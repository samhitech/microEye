import os
import sys
from typing import Any, Optional

from microEye.hardware.pycromanager.headless import HeadlessInstance, HeadlessManager
from microEye.hardware.pycromanager.widgets.headless_options import (
    HeadlessOptions,
    headlessParams,
)
from microEye.qt import QAction, QApplication, Qt, QtCore, QtWidgets, Signal


class HeadlessManagerWidget(QtWidgets.QDialog):
    '''Widget to display and manage running headless instances.'''

    instance_started = Signal(int)  # Signal emitted when new instance is started
    instance_stopped = Signal(int)  # Signal emitted when instance is stopped

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)
        self.options_dialog = None

        self._setup_ui()
        self._connect_signals()
        self._refresh_instances()

        self.setWindowTitle('Micro-Manager Headless Manager')
        self.setMinimumWidth(700)

    def _setup_ui(self):
        '''Set up the UI components.'''
        layout = QtWidgets.QVBoxLayout()

        # Instance list
        self.instance_list = QtWidgets.QTableWidget()
        self.instance_list.setColumnCount(4)
        self.instance_list.setHorizontalHeaderLabels(
            ['Name', 'Port', 'Config', 'Actions']
        )
        self.instance_list.horizontalHeader().setStretchLastSection(True)
        self.instance_list.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.instance_list.setSelectionBehavior(
            QtWidgets.QTableWidget.SelectionBehavior.SelectRows
        )
        layout.addWidget(self.instance_list)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.add_button = QtWidgets.QPushButton('Add Instance')
        self.refresh_button = QtWidgets.QPushButton('Refresh')
        self.stop_all_button = QtWidgets.QPushButton('Stop All')

        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.stop_all_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _connect_signals(self):
        '''Connect signals to slots.'''
        self.add_button.clicked.connect(self._show_options_dialog)
        self.refresh_button.clicked.connect(self._refresh_instances)
        self.stop_all_button.clicked.connect(self._stop_all_instances)

    def _refresh_instances(self):
        '''Refresh the list of running instances.'''
        self.instance_list.setRowCount(0)

        instances = HeadlessManager().instances
        self.instance_list.setRowCount(len(instances))

        for idx, (port, instance) in enumerate(instances.items()):
            # Name
            name_item = QtWidgets.QTableWidgetItem(instance.name)
            self.instance_list.setItem(idx, 0, name_item)

            # Port
            port_item = QtWidgets.QTableWidgetItem(str(instance.port))
            self.instance_list.setItem(idx, 1, port_item)

            # Config file
            config_item = QtWidgets.QTableWidgetItem(
                os.path.basename(instance.config_file)
                if instance.config_file
                else 'None'
            )
            self.instance_list.setItem(idx, 2, config_item)

            # Action buttons
            action_widget = QtWidgets.QWidget()
            action_layout = QtWidgets.QHBoxLayout(action_widget)
            action_layout.setContentsMargins(0, 0, 0, 0)

            edit_button = QtWidgets.QPushButton('Details')
            stop_button = QtWidgets.QPushButton('Stop')

            edit_button.clicked.connect(
                lambda checked, p=port: self._show_instance_details(p)
            )
            stop_button.clicked.connect(lambda checked, p=port: self._stop_instance(p))

            action_layout.addWidget(edit_button)
            action_layout.addWidget(stop_button)

            self.instance_list.setCellWidget(idx, 3, action_widget)

        self.instance_list.resizeRowsToContents()

    def _show_options_dialog(self):
        '''Show dialog to add a new instance.'''
        if not self.options_dialog:
            self.options_dialog = QtWidgets.QDialog(self)
            self.options_dialog.setWindowFlags(
                self.options_dialog.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
            )
            self.options_dialog.setWindowTitle('Add Headless Instance')

            self.options_dialog.setLayout(QtWidgets.QVBoxLayout())

            self.options = HeadlessOptions()
            self.options_dialog.layout().addWidget(self.options)

            # Add buttons to accept/reject
            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )

            button_box.accepted.connect(self._add_instance)
            button_box.rejected.connect(self.options_dialog.close)

            self.options_dialog.layout().addWidget(button_box)

        self.options_dialog.exec()

    def _add_instance(self):
        '''Add a new instance from the options dialog.'''
        getValues = self.options.get_param_value

        # Create instance config from params
        instance_config = HeadlessInstance(
            port=getValues(headlessParams.PORT.label()),
            mm_app_path=getValues(headlessParams.MM_APP_PATH.label()),
            config_file=getValues(headlessParams.CONFIG_FILE.label()),
            java_loc=getValues(headlessParams.JAVA_LOC.label()),
            python_backend=getValues(headlessParams.PYTHON_BACKEND.label()),
            core_log_path=getValues(headlessParams.CORE_LOG_PATH.label()),
            buffer_size_mb=getValues(headlessParams.BUFFER_SIZE_MB.label()),
            max_memory_mb=getValues(headlessParams.MAX_MEMORY_MB.label()),
            debug=getValues(headlessParams.DEBUG.label()),
        )

        # Try to start the instance
        if HeadlessManager().start_instance(instance_config):
            self._refresh_instances()
            self.instance_started.emit(instance_config.port)
            self.options_dialog.close()
        else:
            QtWidgets.QMessageBox.warning(
                self,
                'Failed to Start',
                f'Failed to start instance on port {instance_config.port}.'
                + ' Port may be in use.',
            )

    def _stop_instance(self, port: int):
        '''Stop a specific instance.'''
        if HeadlessManager().stop_instance(port):
            self._refresh_instances()
            self.instance_stopped.emit(port)
        else:
            QtWidgets.QMessageBox.warning(
                self, 'Failed to Stop', f'Failed to stop instance on port {port}.'
            )

    def _stop_all_instances(self):
        '''Stop all running instances.'''
        confirm = QtWidgets.QMessageBox.question(
            self,
            'Confirm Stop All',
            'Are you sure you want to stop all running headless instances?',
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            HeadlessManager().stop_all_instances()
            self._refresh_instances()

    def _show_instance_details(self, port: int):
        '''Show details for a specific instance.'''
        instance = HeadlessManager().get_instance(port)
        if not instance:
            return

        details = QtWidgets.QDialog(self)
        details.setWindowTitle(f'Instance Details: {instance.name}')

        layout = QtWidgets.QVBoxLayout()

        # Create a formatted text display
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)

        detail_text = f'''<h3>Headless Instance Details</h3>
        <p><b>Name:</b> {instance.name}</p>
        <p><b>Port:</b> {instance.port}</p>
        <p><b>Micro-Manager Path:</b> {instance.mm_app_path}</p>
        <p><b>Config File:</b> {instance.config_file or "None"}</p>
        <p><b>Java Location:</b> {instance.java_loc or "Default"}</p>
        <p><b>Python Backend:</b> {"Yes" if instance.python_backend else "No"}</p>
        <p><b>Buffer Size:</b> {instance.buffer_size_mb} MB</p>
        <p><b>Max Memory:</b> {instance.max_memory_mb} MB</p>
        <p><b>Debug Mode:</b> {"Enabled" if instance.debug else "Disabled"}</p>
        '''

        text.setHtml(detail_text)
        layout.addWidget(text)

        # Add a close button
        close_button = QtWidgets.QPushButton('Close')
        close_button.clicked.connect(details.close)
        layout.addWidget(close_button)

        details.setLayout(layout)
        details.resize(500, 400)
        details.exec()

    @classmethod
    def show_dialog(cls, parent: Optional[QtWidgets.QWidget] = None) -> None:
        '''Show the headless manager widget.'''
        if not hasattr(cls, 'singleton'):
            cls.singleton = HeadlessManagerWidget(parent=parent)
        cls.singleton.exec()

    @classmethod
    def get_menu_action(
        cls, parent: Optional[QtWidgets.QWidget] = None
    ) -> QAction:
        '''Get the action to show this widget in a menu.'''
        action = QAction('Micro-Manager Headless Manager', parent=parent)
        action.triggered.connect(lambda: cls.show_dialog(parent=parent))
        action.setStatusTip('Show the Micro-Manager Headless Manager')
        return action


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HeadlessManagerWidget()
    window.show()
    sys.exit(app.exec())
