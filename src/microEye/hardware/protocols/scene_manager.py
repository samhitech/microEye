import threading
import traceback
from typing import Any

from microEye.hardware.protocols.actions import ActionGroup, BaseAction
from microEye.hardware.protocols.actions_items import (
    ActionGroupItem,
    BaseActionItem,
    get_action_item,
)
from microEye.hardware.protocols.serialization import (
    deserialize_action,
    serialize_action,
)
from microEye.qt import Qt, QtCore, QtGui, QtWidgets, Signal
from microEye.utils.thread_worker import QThreadWorker


class SceneManager(QtCore.QObject):
    '''
    Class responsible for managing the scene and its layout.
    '''

    updateTerminal = Signal(str, int)
    clearTerminal = Signal()

    def __init__(self):
        '''
        Initialize a new SceneManager instance.
        '''
        super().__init__()
        self.scene = QtWidgets.QGraphicsScene()
        self.action = ActionGroup()
        self.action.name = 'Main'
        self.root = ActionGroupItem(self.action)
        self.scene.addItem(self.root)

        self.worker = None

    def add_action(self, action: BaseAction):
        '''
        Add an action to the scene.

        Parameters
        ----------
        action : BaseAction
            The action to be added.
        '''
        selected_items = self.scene.selectedItems()

        if len(selected_items) == 1:
            action_item = get_action_item(action)

            if isinstance(selected_items[0], ActionGroupItem):
                selected_item = selected_items[0]
                selected_item.add_child_item(action_item)
            else:
                selected_item = selected_items[0]
                parent_item = selected_item.parentItem()
                if isinstance(parent_item, ActionGroupItem):
                    parent_item.add_child_item(action_item)

    def handleKeyPress(self, event: QtGui.QKeyEvent):
        selected_items = self.scene.selectedItems()

        if len(selected_items) == 1 and isinstance(selected_items[0], BaseActionItem):
            selected_item = selected_items[0]
            parent_item = selected_item.parentItem()
            if parent_item is not None and isinstance(parent_item, ActionGroupItem):
                index_self = parent_item.child_items.index(selected_item)

                if event.key() == Qt.Key.Key_PageUp:
                    # Move item up
                    if index_self > 0:
                        parent_item.swap_children(index_self, index_self - 1)
                elif event.key() == Qt.Key.Key_PageDown:
                    # Move item down
                    if index_self < len(parent_item.child_items) - 1:
                        parent_item.swap_children(index_self, index_self + 1)
                elif event.key() == Qt.Key.Key_Delete:
                    # Delete item and its children
                    parent_item.remove_child_item(selected_item)
                    self.scene.removeItem(selected_item)

    def remove_action(self, action: BaseAction):
        '''
        Remove an action from the scene.

        Parameters
        ----------
        action : BaseAction
            The action to be removed.
        '''
        action_item = self.find_action_item(action)
        if action_item:
            parent_item = action_item.parentItem()
            if parent_item:
                parent_item.remove_child_item(action_item)
            self.scene.removeItem(action_item)

    def find_action_item(self, action: BaseAction) -> ActionGroupItem:
        '''
        Find the action item corresponding to the given action.

        Parameters
        ----------
        action : BaseAction
            The action to find.

        Returns
        -------
        ActionGroupItem
            The action item corresponding to the given action, or None if not found.
        '''
        for item in self.scene.items():
            if isinstance(item, ActionGroupItem) and item.action == action:
                return item
        return None

    def execute_actions(self):
        '''
        Execute the actions in the scene.
        '''
        if isinstance(self.root, ActionGroupItem) and (
            self.worker is None or self.worker.done
        ):
            self.clearTerminal.emit()

            def actions(root: ActionGroupItem, updateTerminal, **kwargs):
                try:
                    root.action.execute(
                        output=updateTerminal, level=0, event=kwargs.get('event')
                    )
                except Exception:
                    updateTerminal.emit(traceback.format_exc(), 0)
                finally:
                    updateTerminal.emit('Done!', 0)

            self.worker = QThreadWorker(
                actions, self.root, self.updateTerminal
            )
            # Execute
            QtCore.QThreadPool.globalInstance().start(self.worker)

    def stop_execution(self):
        if self.worker:
            self.worker.stop()  # This will signal the thread to stop

        # self.updateTerminal.emit('Execution stopped.', 0)

    def load_actions(self, actions: list[dict[str, Any]]):
        '''
        Load actions from a list of serialized action data.

        Parameters
        ----------
        actions : list[dict[str, Any]]
            A list of serialized action data.
        '''
        self.clear_scene()
        for action_data in actions:
            action, action_item = deserialize_action(action_data)
            if isinstance(action_item, ActionGroupItem):
                self.root.add_child_item(action_item)
            else:
                root_child_item = get_action_item(action)
                self.root.add_child_item(root_child_item)

    def clear_scene(self):
        '''
        Clear the scene by removing all items.
        '''
        self.scene.clear()
        BaseAction.id_counter = 1
        self.action = ActionGroup()
        self.action.name = 'Main'
        self.root = ActionGroupItem(self.action)
        self.scene.addItem(self.root)

    def serialize_protocol(self) -> list[dict[str, Any]]:
        '''
        Serialize the protocol to a JSON-serializable data structure.

        Returns
        -------
        list[dict[str, Any]]
            A list of serialized action data.
        '''
        protocol_data = []
        for action in self.root.action.child_actions:
            action_data = serialize_action(action)
            protocol_data.append(action_data)
        return protocol_data
