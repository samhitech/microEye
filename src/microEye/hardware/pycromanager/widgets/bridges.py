import contextlib
import os
import sys
from typing import Any, Optional

from microEye.hardware.pycromanager.core import PycroCore
from microEye.qt import QAction, QApplication, Qt, QtCore, QtWidgets, Signal


class BridgesWidget(QtWidgets.QDialog):
    '''
    Widget to display and manage running PycroCore bridge instances and their devices.
    '''

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

        self.setWindowTitle('Micro-Manager Core Bridge Manager')
        self.setMinimumWidth(900)
        self.setMinimumHeight(600)

    def _setup_ui(self):
        '''Set up the UI components.'''
        main_layout = QtWidgets.QVBoxLayout(self)

        # Splitter for instances and devices
        self.splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)

        # Upper part: Core instances
        instances_widget = QtWidgets.QWidget()
        instances_layout = QtWidgets.QVBoxLayout(instances_widget)

        instances_label = QtWidgets.QLabel('<b>Core Bridge Instances</b>')
        instances_layout.addWidget(instances_label)

        # Instance list
        self.instance_list = QtWidgets.QTableWidget()
        self.instance_list.setColumnCount(4)
        self.instance_list.setHorizontalHeaderLabels(
            ['Port', 'Status', 'Devices', 'Actions']
        )
        self.instance_list.horizontalHeader().setStretchLastSection(True)
        self.instance_list.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.instance_list.setSelectionBehavior(
            QtWidgets.QTableWidget.SelectionBehavior.SelectRows
        )
        self.instance_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.instance_list.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        instances_layout.addWidget(self.instance_list)

        # Lower part: Device details
        devices_widget = QtWidgets.QWidget()
        devices_layout = QtWidgets.QVBoxLayout(devices_widget)

        devices_label = QtWidgets.QLabel('<b>Device Details</b>')
        devices_layout.addWidget(devices_label)

        # Device list for selected core
        self.device_list = QtWidgets.QTableWidget()
        self.device_list.setColumnCount(3)
        self.device_list.setHorizontalHeaderLabels(['Label', 'Category', 'Properties'])
        self.device_list.horizontalHeader().setStretchLastSection(True)
        self.device_list.setSelectionBehavior(
            QtWidgets.QTableWidget.SelectionBehavior.SelectRows
        )
        self.device_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.device_list.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.device_list.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        devices_layout.addWidget(self.device_list)

        # Add widgets to splitter
        self.splitter.addWidget(instances_widget)
        self.splitter.addWidget(devices_widget)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.splitter)

        # Bottom buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.add_button = QtWidgets.QPushButton('Add Core Bridge')
        self.refresh_button = QtWidgets.QPushButton('Refresh')
        self.stop_all_button = QtWidgets.QPushButton('Stop All')

        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.stop_all_button)
        # button_layout.addStretch()

        main_layout.addLayout(button_layout)

    def _connect_signals(self):
        '''Connect signals to slots.'''
        self.add_button.clicked.connect(self._show_add_bridge_dialog)
        self.refresh_button.clicked.connect(self._refresh_instances)
        self.stop_all_button.clicked.connect(self._stop_all_instances)
        self.instance_list.itemSelectionChanged.connect(self._update_device_list)

    def _refresh_instances(self):
        '''Refresh the list of running core bridge instances.'''
        self.instance_list.setRowCount(0)
        self.device_list.setRowCount(0)

        instances = PycroCore._instances
        if not instances:
            # Show message if no instances
            self.instance_list.setRowCount(1)
            no_instances = QtWidgets.QTableWidgetItem('No active core bridges')
            self.instance_list.setSpan(0, 0, 1, 4)
            self.instance_list.setItem(0, 0, no_instances)
            return

        self.instance_list.setRowCount(len(instances))

        for idx, (port, core_instance) in enumerate(instances.items()):
            # Port
            port_item = QtWidgets.QTableWidgetItem(str(port))
            self.instance_list.setItem(idx, 0, port_item)

            # Status
            status = core_instance.is_connected()
            status_text = 'Connected' if status else 'Disconnected'
            status_item = QtWidgets.QTableWidgetItem(status_text)
            self.instance_list.setItem(idx, 1, status_item)

            # Device count
            device_count = 0
            try:
                if status:
                    device_count = len(core_instance.get_loaded_devices())
            except Exception:
                pass

            device_item = QtWidgets.QTableWidgetItem(f'{device_count} devices')
            self.instance_list.setItem(idx, 2, device_item)

            stop_button = QtWidgets.QPushButton('Stop')

            stop_button.clicked.connect(
                lambda checked=False, p=port: self._stop_instance(p)
            )

            self.instance_list.setCellWidget(idx, 3, stop_button)

        self.instance_list.resizeRowsToContents()

        # Auto-select first instance
        if instances:
            self.instance_list.selectRow(0)

    def _update_device_list(self):
        '''Update the device list based on selected core instance.'''
        self.device_list.setRowCount(0)

        selected_rows = self.instance_list.selectionModel().selectedRows()
        if not selected_rows:
            return

        # Get the port from the first column of the selected row
        selected_row = selected_rows[0].row()
        port_item = self.instance_list.item(selected_row, 0)

        if not port_item:
            return

        if not port_item.text().isdigit():
            return

        try:
            port = int(port_item.text())
            core_instance = PycroCore._instances.get(port)

            if not core_instance or not core_instance.is_connected():
                return

            devices = core_instance.get_loaded_devices()
            self.device_list.setRowCount(len(devices))

            for idx, device in enumerate(devices):
                # Device Label
                name_item = QtWidgets.QTableWidgetItem(device)
                self.device_list.setItem(idx, 0, name_item)

                # Device Category
                try:
                    category = core_instance.get_device_type(device).name
                except Exception:
                    category = 'Unknown'
                category_item = QtWidgets.QTableWidgetItem(category)
                self.device_list.setItem(idx, 1, category_item)

                # Property count
                try:
                    props = core_instance.get_device_property_names(device)
                    prop_count = len(props)
                except Exception:
                    prop_count = 0

                props_button = QtWidgets.QPushButton(f'Properties ({prop_count})')
                props_button.setToolTip('Show device properties')
                props_button.clicked.connect(
                    lambda checked=False,
                    dev=device,
                    core=core_instance: self._show_device_properties(dev, core)
                )
                self.device_list.setCellWidget(idx, 2, props_button)

            self.device_list.resizeRowsToContents()

        except Exception as e:
            print(f'Error updating device list: {e}')
            return

    def _show_device_properties(self, device_name: str, core_instance: PycroCore):
        '''Show properties for a specific device.'''
        try:
            props = core_instance.get_device_property_names(device_name)

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(f'Properties: {device_name}')
            dialog.setMinimumWidth(500)
            dialog.setMinimumHeight(400)

            layout = QtWidgets.QVBoxLayout(dialog)

            prop_table = QtWidgets.QTableWidget()
            prop_table.setColumnCount(3)
            prop_table.setHorizontalHeaderLabels(['Property', 'Value', 'Readonly'])
            prop_table.setRowCount(len(props))

            for idx, prop in enumerate(props):
                # Property name
                prop_item = QtWidgets.QTableWidgetItem(prop)
                prop_table.setItem(idx, 0, prop_item)

                # Property value
                try:
                    value = core_instance.get_property(device_name, prop)
                except Exception:
                    value = 'Error'
                value_item = QtWidgets.QTableWidgetItem(str(value))
                prop_table.setItem(idx, 1, value_item)

                # Property readonly status
                try:
                    readonly = core_instance.is_property_readonly(device_name, prop)
                    readonly_text = 'Yes' if readonly else 'No'
                except Exception:
                    readonly_text = 'Unknown'
                readonly_item = QtWidgets.QTableWidgetItem(readonly_text)
                prop_table.setItem(idx, 2, readonly_item)

            prop_table.horizontalHeader().setStretchLastSection(True)
            prop_table.horizontalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )

            layout.addWidget(prop_table)

            close_button = QtWidgets.QPushButton('Close')
            close_button.clicked.connect(dialog.close)
            layout.addWidget(close_button)

            dialog.exec()

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, 'Error', f'Failed to get properties: {str(e)}'
            )

    def _show_add_bridge_dialog(self):
        '''Show dialog to add a new core bridge instance.'''
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Add Core Bridge')
        layout = QtWidgets.QFormLayout(dialog)

        port_input = QtWidgets.QSpinBox()
        port_input.setRange(1024, 65535)
        port_input.setValue(4827)  # Default MM port
        layout.addRow('Port:', port_input)

        # Add connect button
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )

        def add_instance():
            port = port_input.value()
            try:
                # Try to create a new core instance
                core = PycroCore(port=port)
                if core.is_connected():
                    self._refresh_instances()
                    dialog.accept()
                    self.instance_started.emit(port)
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        'Connection Failed',
                        'Could not connect to Micro-Manager core at the specified port',
                    )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, 'Error', f'Failed to connect: {str(e)}'
                )

        button_box.accepted.connect(add_instance)
        button_box.rejected.connect(dialog.reject)

        layout.addRow(button_box)
        dialog.exec()

    def _stop_instance(self, port: int):
        '''Stop a specific core bridge instance.'''
        confirm = QtWidgets.QMessageBox.question(
            self,
            f'Confirm Stop {port}',
            f'Are you sure you want to stop the instance on port {port}?'
            + '\nThis will close the connection to the Micro-Manager core.',
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return False

        try:
            core_instance = PycroCore._instances.get(port)
            if core_instance:
                core_instance.close()
                if port in PycroCore._instances:
                    del PycroCore._instances[port]
                self._refresh_instances()
                self.instance_stopped.emit(port)
                return True
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, 'Error', f'Failed to stop instance: {str(e)}'
            )
        return False

    def _stop_all_instances(self):
        '''Stop all running core bridge instances.'''
        confirm = QtWidgets.QMessageBox.question(
            self,
            'Confirm Stop All',
            'Are you sure you want to stop all running core bridges?',
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        instances = list(PycroCore._instances.keys())
        for port in instances:
            with contextlib.suppress(Exception):
                self._stop_instance(port)
        self._refresh_instances()

    @classmethod
    def show_dialog(cls, parent: Optional[QtWidgets.QWidget] = None) -> None:
        '''Show the bridges manager widget as a dialog.'''
        if not hasattr(cls, '_singleton'):
            cls._singleton = cls(parent=parent)

        cls._singleton.exec()

    @classmethod
    def get_menu_action(cls, parent: Optional[QtWidgets.QWidget] = None) -> QAction:
        '''Get the action to show this widget in a menu.'''
        action = QAction('Micro-Manager Core Bridges', parent=parent)
        action.triggered.connect(lambda: cls.show_dialog(parent=parent))
        action.setStatusTip('Show the Micro-Manager Core Bridge Manager')
        return action


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BridgesWidget()
    window.show()
    sys.exit(app.exec())
