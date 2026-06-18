import json
import logging
import os

import pyqtgraph as pg

from microEye.analysis.tools.common import normalize_positive_float, to_float_or_none
from microEye.analysis.tools.dark_cal.analysis import (
    has_plot_data,
    perform_analysis,
    resolve_results_mode,
)
from microEye.analysis.tools.dark_cal.compare import DarkDatasetComparisonDialog
from microEye.analysis.tools.dark_cal.constants import ABOUT_HTML, FILE_NAMES, DataTypes
from microEye.analysis.tools.dark_cal.io import load_results_npz, save_results_npz
from microEye.analysis.tools.dark_cal.plotting import (
    plot_results,
    plot_results_matplotlib,
)
from microEye.qt import (
    QtCore,
    QtWidgets,
    getExistingDirectory,
    getOpenFileName,
    getSaveFileName,
)
from microEye.utils.thread_worker import QThreadWorker

logger = logging.getLogger(__name__)


class DarkCalibration(QtWidgets.QDialog):
    NAME = 'Dark Calibration Tool'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.NAME)
        self.resize(1200, 800)

        self._directories = {}
        self._dataset_meta: dict[str, dict] = {}
        self._plot_widgets: list[pg.GraphicsLayoutWidget] = []

        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 2)

        tools = QtWidgets.QWidget()
        tabs.addTab(tools, 'Tools')

        vertical_layout = QtWidgets.QVBoxLayout()
        tools.setLayout(vertical_layout)

        directories = QtWidgets.QGroupBox('Datasets')
        vertical_layout.addWidget(directories)

        directories_layout = QtWidgets.QVBoxLayout()
        directories.setLayout(directories_layout)

        self.datasets_table = QtWidgets.QTableWidget()
        self.datasets_table.setColumnCount(6)
        self.datasets_table.setHorizontalHeaderLabels(
            [
                'Name',
                'Gain (e-/ADU)',
                'Gain Source',
                'Responsivity (ADU/photon)',
                'QE',
                'Dark Cal Directory',
            ]
        )
        self.datasets_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.datasets_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.datasets_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        directories_layout.addWidget(self.datasets_table)

        button_layout = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton('Add Directory')
        self.import_button = QtWidgets.QPushButton('Import Datasets')
        self.import_button.setToolTip('Import datasets and metadata from JSON file')
        self.export_button = QtWidgets.QPushButton('Export Datasets')
        self.export_button.setToolTip('Export dataset list and metadata to JSON file')
        self.remove_button = QtWidgets.QPushButton('Remove Selected')
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.remove_button)
        directories_layout.addLayout(button_layout)

        self.options_groupbox = QtWidgets.QGroupBox('Analysis Options')
        vertical_layout.addWidget(self.options_groupbox)

        options_layout = QtWidgets.QFormLayout()
        self.options_groupbox.setLayout(options_layout)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Standard', 'Histograms'])
        options_layout.addRow('Analysis Mode:', self.mode)

        self.run_button = QtWidgets.QPushButton('Run Analysis')
        self.compare_button = QtWidgets.QPushButton('Compare Datasets')
        self.matplot_button = QtWidgets.QPushButton('Open plotter')
        options_layout.addWidget(self.run_button)
        options_layout.addWidget(self.compare_button)
        options_layout.addWidget(self.matplot_button)

        cache_buttons_widget = QtWidgets.QWidget()
        cache_buttons_layout = QtWidgets.QHBoxLayout(cache_buttons_widget)
        cache_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.load_results_button = QtWidgets.QPushButton('Load Results')
        self.save_results_button = QtWidgets.QPushButton('Save Results')
        cache_buttons_layout.addWidget(self.load_results_button)
        cache_buttons_layout.addWidget(self.save_results_button)
        options_layout.addRow('Results Cache:', cache_buttons_widget)

        tabs_groupbox = QtWidgets.QGroupBox('Analysis Results')
        layout.addWidget(tabs_groupbox, 3)

        self.tab_widget = QtWidgets.QTabWidget()
        tabs_groupbox.setLayout(QtWidgets.QVBoxLayout())
        tabs_groupbox.layout().addWidget(self.tab_widget)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        tabs_groupbox.layout().addWidget(self.progress_bar)

        tabs.addTab(self._setup_about(), 'About')

        self.add_button.clicked.connect(self.add_directory_clicked)
        self.import_button.clicked.connect(self.import_datasets_clicked)
        self.export_button.clicked.connect(self.export_datasets_clicked)
        self.remove_button.clicked.connect(self.remove_selected_directories)
        self.run_button.clicked.connect(self.run_analysis)
        self.compare_button.clicked.connect(self.compare_datasets)
        self.load_results_button.clicked.connect(self.import_results)
        self.save_results_button.clicked.connect(self.export_results)
        self.matplot_button.clicked.connect(self.open_plotter)

    def _setup_about(self):
        about_widget = QtWidgets.QTextEdit()
        about_widget.setHtml(ABOUT_HTML)
        about_widget.setReadOnly(True)
        return about_widget

    def _set_controls_enabled(self, enabled: bool):
        self.add_button.setEnabled(enabled)
        self.import_button.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.remove_button.setEnabled(enabled)
        self.run_button.setEnabled(enabled)
        self.compare_button.setEnabled(enabled)
        self.load_results_button.setEnabled(enabled)
        self.save_results_button.setEnabled(enabled)
        self.matplot_button.setEnabled(enabled)

    def _default_dataset_meta(self, directory: str) -> dict:
        base_name = os.path.basename(os.path.normpath(str(directory))) or str(directory)
        return {
            'name': base_name,
            'dark_calibration_directory': str(directory),
            'gain': 1.0,
            'gain_defaulted': True,
            'responsivity': None,
            'quantum_efficiency': None,
        }

    def _normalize_gain(self, value) -> tuple[float, bool]:
        return normalize_positive_float(value, default=1.0)

    def _resolve_import_directory(
        self,
        dataset_name: str,
        meta: dict,
        root: str,
    ) -> str:
        directory = (
            meta.get('dark_calibration_directory')
            or meta.get('dark_cal')
            or meta.get('directory')
            or meta.get('path')
        )

        if not directory:
            signal_path = meta.get('signal')
            dark_path = meta.get('dark')
            if signal_path:
                directory = os.path.dirname(str(signal_path))
            elif dark_path:
                directory = os.path.dirname(str(dark_path))
            else:
                directory = dataset_name

        directory = str(directory)
        if not os.path.isabs(directory):
            directory = os.path.join(root, directory)

        return os.path.normpath(directory)

    def _add_directory(self, directory: str, parent=None):
        if directory and directory not in self._directories:
            if any(
                not os.path.exists(os.path.join(directory, FILE_NAMES[data_type]))
                for data_type in [
                    DataTypes.MEAN,
                    DataTypes.VARIANCE,
                    DataTypes.EXPOSURE,
                ]
            ):
                if parent is None:
                    for root, dirs, _ in os.walk(directory):
                        for d in dirs:
                            self._add_directory(os.path.join(root, d), parent=directory)

                return

            self._directories[directory] = {}
            self._dataset_meta[directory] = self._default_dataset_meta(directory)

    def import_datasets_clicked(self):
        json_path, _ = getOpenFileName(
            self, 'Import Datasets JSON', '', 'JSON (*.json)'
        )
        if not json_path:
            return

        try:
            with open(json_path, encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as e:
            logger.error(f'Failed to read JSON: {e}')
            return

        if not isinstance(payload, dict):
            logger.error('Invalid JSON format: expected top-level object/dict.')
            return

        if isinstance(payload.get('datasets'), dict):
            entries = payload['datasets']
            root = str(
                payload.get('dark_cal_root')
                or payload.get('data_root')
                or os.path.dirname(json_path)
            )
        else:
            entries = payload
            root = os.path.dirname(json_path)

        added = 0
        skipped = 0
        defaulted_gain = 0

        for dataset_name, meta in entries.items():
            try:
                if not isinstance(meta, dict):
                    raise ValueError('Entry must be an object/dict')

                directory = self._resolve_import_directory(
                    str(dataset_name),
                    meta,
                    root,
                )
                if not os.path.isdir(directory):
                    raise FileNotFoundError(f'Directory not found: {directory}')

                gain, gain_defaulted = self._normalize_gain(meta.get('gain'))
                metadata = {
                    'name': str(meta.get('name', dataset_name)),
                    'dark_calibration_directory': directory,
                    'gain': gain,
                    'gain_defaulted': gain_defaulted,
                    'responsivity': to_float_or_none(meta.get('responsivity')),
                    'quantum_efficiency': to_float_or_none(
                        meta.get('quantum_efficiency')
                    ),
                }

                before_count = len(self._directories)
                self._add_directory(directory)
                if directory not in self._directories:
                    raise ValueError(
                        'Missing required dark calibration files in directory'
                    )

                self._dataset_meta[directory] = metadata

                if len(self._directories) > before_count:
                    added += 1
                if gain_defaulted:
                    defaulted_gain += 1
            except Exception as e:
                skipped += 1
                logger.warning(f'Skipped "{dataset_name}": {e}')

        self.update_directory_list()
        logger.info(
            f'Imported {added} dataset(s) from {os.path.basename(json_path)}; '
            f'skipped {skipped}; defaulted gain for {defaulted_gain} dataset(s).'
        )

    def export_datasets_clicked(self):
        if not self._directories:
            logger.info('No datasets to export.')
            return

        json_path, _ = getSaveFileName(
            self, 'Export Datasets JSON', '', 'JSON (*.json)'
        )
        if not json_path:
            return

        if not json_path.lower().endswith('.json'):
            json_path = f'{json_path}.json'

        payload = {
            'dark_cal_root': os.path.dirname(json_path),
            'datasets': {
                meta.get('name', os.path.basename(directory)): {
                    'name': meta.get('name'),
                    'dark_calibration_directory': directory,
                    'gain': meta.get('gain'),
                    'gain_defaulted': meta.get('gain_defaulted'),
                    'responsivity': meta.get('responsivity'),
                    'quantum_efficiency': meta.get('quantum_efficiency'),
                }
                for directory, meta in self._dataset_meta.items()
            },
        }

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=4)
            logger.info(f'Exported {len(self._directories)} dataset(s) to {json_path}')
        except Exception as e:
            logger.error(f'Failed to write JSON: {e}')

    def add_directory_clicked(self):
        directory = getExistingDirectory(self, 'Select Directory')
        self._add_directory(directory)
        self.update_directory_list()

    def remove_selected_directories(self):
        selected_rows = {item.row() for item in self.datasets_table.selectedItems()}

        for row in sorted(selected_rows, reverse=True):
            name_item = self.datasets_table.item(row, 0)
            if name_item is None:
                continue

            directory = name_item.data(QtCore.Qt.ItemDataRole.UserRole)
            if directory in self._directories:
                del self._directories[directory]
            if directory in self._dataset_meta:
                del self._dataset_meta[directory]

        self.update_directory_list()

    def update_directory_list(self):
        self.datasets_table.clearContents()
        self.datasets_table.setRowCount(0)

        for directory in self._directories:
            meta = self._dataset_meta.get(
                directory,
                self._default_dataset_meta(directory),
            )

            row = self.datasets_table.rowCount()
            self.datasets_table.insertRow(row)

            gain = float(meta.get('gain', 1.0))
            gain_defaulted = bool(meta.get('gain_defaulted', True))
            name = str(meta.get('name', os.path.basename(directory)))

            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, directory)
            name_item.setToolTip(directory)
            self.datasets_table.setItem(row, 0, name_item)

            self.datasets_table.setItem(
                row,
                1,
                QtWidgets.QTableWidgetItem(f'{gain:.6g}'),
            )

            gain_source = 'Default (1.0)' if gain_defaulted else 'Provided'
            self.datasets_table.setItem(
                row,
                2,
                QtWidgets.QTableWidgetItem(gain_source),
            )

            responsivity = meta.get('responsivity')
            responsivity_value = to_float_or_none(responsivity)
            responsivity_text = (
                'N/A' if responsivity_value is None else f'{responsivity_value:.6g}'
            )
            self.datasets_table.setItem(
                row,
                3,
                QtWidgets.QTableWidgetItem(responsivity_text),
            )

            qe = meta.get('quantum_efficiency')
            qe_value = to_float_or_none(qe)
            qe_text = 'N/A' if qe_value is None else f'{qe_value:.6g}'
            self.datasets_table.setItem(row, 4, QtWidgets.QTableWidgetItem(qe_text))

            directory_item = QtWidgets.QTableWidgetItem(directory)
            directory_item.setToolTip(directory)
            self.datasets_table.setItem(row, 5, directory_item)

    def clear_plots(self):
        for widget in self._plot_widgets:
            widget.close()
        self._plot_widgets = []

    def _plot_ready_directories(self) -> dict:
        return {
            directory: data
            for directory, data in self._directories.items()
            if has_plot_data(data)
        }

    def export_results(self):
        plot_ready = self._plot_ready_directories()
        if not plot_ready:
            logger.info('No analysis results to export. Run analysis first.')
            return

        try:
            mode = resolve_results_mode(plot_ready)
        except Exception as e:
            logger.error(f'Failed to determine results mode: {e}')
            return

        if not mode:
            logger.info('No plot-ready results to export.')
            return

        file_path, _ = getSaveFileName(
            self,
            'Export Dark Calibration Results',
            '',
            'NumPy Files (*.npz)',
        )
        if not file_path:
            return

        if not file_path.lower().endswith('.npz'):
            file_path = f'{file_path}.npz'

        try:
            count = save_results_npz(
                file_path,
                plot_ready,
                mode,
                dataset_meta=self._dataset_meta,
            )
            logger.info(f'Exported {count} dataset(s) to {file_path}')
        except Exception as e:
            logger.error(f'Failed to export results: {e}')

    def import_results(self):
        file_path, _ = getOpenFileName(
            self,
            'Load Dark Calibration Results',
            '',
            'NumPy Files (*.npz)',
        )
        if not file_path:
            return

        try:
            mode, directories, loaded_meta = load_results_npz(file_path)
        except Exception as e:
            logger.error(f'Failed to load results: {e}')
            return

        if not directories:
            logger.info('Loaded file contains no datasets.')
            return

        self._directories = directories
        self._dataset_meta = {}
        for directory in directories:
            cached_meta = loaded_meta.get(directory, {})
            merged = self._default_dataset_meta(directory)
            merged.update(cached_meta)
            self._dataset_meta[directory] = merged
        self.update_directory_list()

        mode_index = self.mode.findText(mode)
        if mode_index >= 0:
            self.mode.setCurrentIndex(mode_index)

        self._plot_widgets = plot_results(self._directories, self._dataset_meta, mode)
        self.update_plots(self._plot_widgets)
        self.progress_bar.setValue(100)

        logger.info(f'Loaded {len(directories)} dataset(s) from {file_path}')

    def compare_datasets(self):
        plot_ready = self._plot_ready_directories()
        if len(plot_ready) < 2:
            logger.info('Need at least two analyzed datasets to compare.')
            return

        try:
            mode = resolve_results_mode(plot_ready)
        except Exception as e:
            logger.error(f'Failed to determine comparison mode: {e}')
            return

        if not mode:
            logger.info('No comparable dark calibration results found.')
            return

        dataset_meta = {
            directory: self._dataset_meta.get(
                directory,
                self._default_dataset_meta(directory),
            )
            for directory in plot_ready
        }

        dialog = DarkDatasetComparisonDialog(
            plot_ready,
            mode,
            dataset_meta=dataset_meta,
            parent=self,
        )
        dialog.show()

    def update_plots(self, plot_widgets: list[pg.GraphicsLayoutWidget]):
        self.tab_widget.clear()
        for widget in plot_widgets:
            self.tab_widget.addTab(widget, widget.windowTitle())

    def run_analysis(self):
        if not self._directories:
            logger.info(
                'No Directories | Please add at least one directory to analyze.'
            )
            return

        self._set_controls_enabled(False)
        self.progress_bar.setValue(0)

        def analysis_finished():
            self.progress_bar.setValue(100)
            logger.info('Analysis complete.')

            self._set_controls_enabled(True)

            self._plot_widgets = plot_results(
                self._directories, self._dataset_meta, self.mode.currentText()
            )
            self.update_plots(self._plot_widgets)

        def update_progress(value):
            if isinstance(value, str):
                logger.info(value)
            elif isinstance(value, (int, float)):
                self.progress_bar.setValue(int(value))

        worker = QThreadWorker(
            perform_analysis,
            self._directories,
            self.mode.currentText(),
            progress=True,
        )
        worker.signals.progress.connect(update_progress)
        worker.signals.finished.connect(analysis_finished)

        QtCore.QThreadPool.globalInstance().start(worker)

    def open_plotter(self):
        plot_ready = self._plot_ready_directories()
        if not plot_ready:
            logger.info('No analysis results to plot. Run analysis first.')
            return

        try:
            mode = resolve_results_mode(plot_ready)
        except Exception as e:
            logger.error(f'Failed to determine plot mode: {e}')
            return

        if not mode:
            logger.info('No plot-ready results to display.')
            return

        self._plotter = plot_results_matplotlib(
            plot_ready, self._dataset_meta, mode
        )
        self._plotter.show()


if __name__ == '__main__':
    import sys

    from microEye.qt import QApplication

    app = QApplication(sys.argv)

    reg_widget = DarkCalibration()
    reg_widget.show()

    sys.exit(app.exec())
