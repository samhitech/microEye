import json
import logging
import os

import numpy as np
import pandas as pd
import pyqtgraph as pg

from microEye.analysis.tools.common import to_float_or_none
from microEye.analysis.tools.photon_cal.analysis import perform_analysis
from microEye.analysis.tools.photon_cal.compare import CacheComparisonDialog
from microEye.analysis.tools.photon_cal.constants import ABOUT_HTML
from microEye.analysis.tools.photon_cal.io import (
    load_analysis_cache,
    save_analysis_cache,
)
from microEye.analysis.tools.photon_cal.models import CalibrationDatasetMeta
from microEye.analysis.tools.photon_cal.plotting import plot_results
from microEye.analysis.tools.photon_cal.table import PandasModel, results_to_dataframe
from microEye.analysis.tools.sphere_cal import SphereCalibrationTool
from microEye.qt import (
    QtCore,
    QtWidgets,
    getOpenFileName,
    getSaveFileName,
)
from microEye.utils.thread_worker import QThreadWorker

logger = logging.getLogger(__name__)


class PhotonTransfer(QtWidgets.QDialog):
    NAME = 'Photon Transfer Curve Analysis Tool'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.NAME)
        self.resize(1200, 800)

        self._datasets: dict[str, CalibrationDatasetMeta] = {}
        self._plot_widgets: list[pg.PlotWidget] = []
        self._results_table: pd.DataFrame | None = None
        self._returned_data: dict | None = None
        self._exposure_times_s: np.ndarray | None = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 2)

        general = QtWidgets.QWidget()
        tabs.addTab(general, 'General')

        self.sphere_calibration_tool = SphereCalibrationTool()
        tabs.addTab(self.sphere_calibration_tool, 'Sphere Power Calibration')

        tabs.addTab(self._setup_about(), 'About')

        vertical_layout = QtWidgets.QVBoxLayout()
        general.setLayout(vertical_layout)

        directories = QtWidgets.QGroupBox('Datasets')
        vertical_layout.addWidget(directories)

        directories_layout = QtWidgets.QVBoxLayout()
        directories.setLayout(directories_layout)

        self.datasets_table = QtWidgets.QTableWidget()
        self.datasets_table.setColumnCount(5)
        self.datasets_table.setHorizontalHeaderLabels(
            [
                'Name',
                'Wavelength (nm)',
                'Power (mW)',
                'Port Power (mW)',
                'Pixel Size (µm)',
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
        self.add_button = QtWidgets.QPushButton('Add Dataset')
        self.load_datasets_button = QtWidgets.QPushButton('Import Datasets')
        self.remove_button = QtWidgets.QPushButton('Remove Selected')

        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.load_datasets_button)
        button_layout.addWidget(self.remove_button)

        directories_layout.addLayout(button_layout)

        self.options_groupbox = QtWidgets.QGroupBox('Analysis Options')
        vertical_layout.addWidget(self.options_groupbox)

        options_layout = QtWidgets.QFormLayout()
        self.options_groupbox.setLayout(options_layout)

        self.gain_variance_source = QtWidgets.QComboBox()
        self.gain_variance_source.addItems(
            [
                'Shot variance (var_total - var_dark)',
                'Total variance (var_total)',
            ]
        )
        options_layout.addRow('Gain Fit Variance:', self.gain_variance_source)

        self.run_button = QtWidgets.QPushButton('Run Analysis')
        self.load_cache_button = QtWidgets.QPushButton('Load Analysis Cache')
        self.save_cache_button = QtWidgets.QPushButton('Save Analysis Cache')
        self.export_dark_cal_json_button = QtWidgets.QPushButton(
            'Export Dark Cal JSON'
        )
        self.compare_cache_button = QtWidgets.QPushButton('Compare Two Caches')

        options_layout.addWidget(self.run_button)
        options_layout.addWidget(self.load_cache_button)
        options_layout.addWidget(self.save_cache_button)
        options_layout.addWidget(self.export_dark_cal_json_button)
        options_layout.addWidget(self.compare_cache_button)

        tabs_groupbox = QtWidgets.QGroupBox('Analysis Results')
        layout.addWidget(tabs_groupbox, 5)

        self.tab_widget = QtWidgets.QTabWidget()
        tabs_groupbox.setLayout(QtWidgets.QVBoxLayout())
        tabs_groupbox.layout().addWidget(self.tab_widget)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        tabs_groupbox.layout().addWidget(self.progress_bar)

        self.add_button.clicked.connect(self.add_dataset_clicked)
        self.remove_button.clicked.connect(self.remove_selected_datasets)
        self.load_datasets_button.clicked.connect(self.import_datasets_clicked)
        self.run_button.clicked.connect(self.run_analysis)
        self.load_cache_button.clicked.connect(self.load_analysis_cache)
        self.save_cache_button.clicked.connect(self.save_analysis_cache)
        self.export_dark_cal_json_button.clicked.connect(
            self.export_dark_calibration_json
        )
        self.compare_cache_button.clicked.connect(self.compare_caches)

    def _setup_about(self):
        about_widget = QtWidgets.QTextEdit()
        about_widget.setHtml(ABOUT_HTML)
        about_widget.setReadOnly(True)
        return about_widget

    def _selected_gain_variance_source(self) -> str:
        return 'total' if self.gain_variance_source.currentIndex() == 1 else 'shot'

    def _set_gain_variance_source(self, source: str | None):
        self.gain_variance_source.setCurrentIndex(
            1 if str(source).lower() == 'total' else 0
        )

    def _set_controls_enabled(self, enabled: bool):
        self.add_button.setEnabled(enabled)
        self.load_datasets_button.setEnabled(enabled)
        self.remove_button.setEnabled(enabled)
        self.run_button.setEnabled(enabled)
        self.gain_variance_source.setEnabled(enabled)
        self.load_cache_button.setEnabled(enabled)
        self.save_cache_button.setEnabled(enabled)
        self.export_dark_cal_json_button.setEnabled(enabled)
        self.compare_cache_button.setEnabled(enabled)

    def _sync_dataset_calibration_fields_from_results(self):
        if self._returned_data is None:
            return

        for name, ds in self._datasets.items():
            params = self._returned_data.get(name, {})
            if not isinstance(params, dict):
                continue

            ds.gain = to_float_or_none(params.get('gain_e_per_dn'))
            ds.responsivity = to_float_or_none(params.get('responsivity'))
            ds.quantum_efficiency = to_float_or_none(params.get('qe'))

    def _add_dataset(self, ds: CalibrationDatasetMeta) -> bool:
        if not os.path.exists(ds.signal_path):
            logger.warning(f'Signal file not found: {ds.signal_path}')
            return False
        if not os.path.exists(ds.dark_path):
            logger.warning(f'Dark file not found: {ds.dark_path}')
            return False
        self._datasets[ds.name] = ds
        return True

    def _dataset_from_json_entry(
        self, name: str, meta: dict, wavelength_nm: float
    ) -> CalibrationDatasetMeta:
        signal = meta.get('signal')
        dark = meta.get('dark')
        if not signal or not dark:
            raise ValueError('Missing required keys: signal/dark')

        if not os.path.exists(signal):
            raise FileNotFoundError(f'Signal file not found: {signal}')

        if not os.path.exists(dark):
            raise FileNotFoundError(f'Dark file not found: {dark}')

        port_power_mW = meta.get('port_power')

        laser_power_mW = None
        if port_power_mW is None and meta.get('power') is not None:
            laser_power_mW = float(meta['power'])

        return CalibrationDatasetMeta(
            name=name,
            signal_path=os.path.normpath(str(signal)),
            dark_path=os.path.normpath(str(dark)),
            laser_power_mW=laser_power_mW,
            port_power_mW=None if port_power_mW is None else float(port_power_mW),
            pixel_size_um=(
                None
                if meta.get('pixel_size_um') is None
                else float(meta['pixel_size_um'])
            ),
            wavelength_nm=float(meta.get('wavelength_nm', wavelength_nm)),
            min_exposure_s=(
                None
                if meta.get('min_exposure_s') is None
                else float(meta['min_exposure_s'])
            ),
        )

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

        wavelength_nm, ok = QtWidgets.QInputDialog.getDouble(
            self,
            'Select Wavelength',
            'Wavelength (nm):',
            value=488.0,
        )
        if not ok:
            return

        added = 0
        skipped = 0
        for name, meta in payload.items():
            try:
                if not isinstance(meta, dict):
                    raise ValueError('Entry must be an object/dict')
                ds = self._dataset_from_json_entry(name, meta, wavelength_nm)
                if self._add_dataset(ds):
                    added += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                logger.warning(f'Skipped "{name}": {e}')

        self.update_directory_list()
        logger.info(
            f'Imported {added} dataset(s) from {os.path.basename(json_path)}; '
            f'skipped {skipped}.'
        )

    def add_dataset_clicked(self):
        signal_path, _ = getOpenFileName(
            self, 'Select Signal .npy', '', 'NumPy (*.npy)'
        )
        if not signal_path:
            return

        dark_path, _ = getOpenFileName(
            self, 'Select Dark .npy', os.path.dirname(signal_path), 'NumPy (*.npy)'
        )
        if not dark_path:
            return

        default_name = os.path.splitext(os.path.basename(signal_path))[0]
        name, ok = QtWidgets.QInputDialog.getText(
            self, 'Dataset Name', 'Name:', text=default_name
        )
        if not ok or not name.strip():
            return

        pixel_size_um, ok = QtWidgets.QInputDialog.getDouble(
            self,
            'Pixel Size',
            'Pixel Size (µm):',
            value=6.5,
        )
        if not ok:
            return

        port_power, ok = QtWidgets.QInputDialog.getDouble(
            self,
            'Port Power',
            'Port power (mW):',
            value=0.1,
        )
        if not ok:
            return

        ds = CalibrationDatasetMeta(
            name=name.strip(),
            signal_path=os.path.normpath(signal_path),
            dark_path=os.path.normpath(dark_path),
            pixel_size_um=pixel_size_um,
            port_power_mW=port_power,
        )
        self._add_dataset(ds)
        self.update_directory_list()

    def remove_selected_datasets(self):
        selected_rows = set()
        for item in self.datasets_table.selectedItems():
            selected_rows.add(item.row())

        for row in sorted(selected_rows, reverse=True):
            name = self.datasets_table.item(row, 0).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            if name in self._datasets:
                del self._datasets[name]
        self.update_directory_list()

    def update_directory_list(self):
        self.datasets_table.clearContents()
        self.datasets_table.setRowCount(0)

        for name, ds in self._datasets.items():
            row = self.datasets_table.rowCount()
            self.datasets_table.insertRow(row)

            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            self.datasets_table.setItem(row, 0, name_item)

            wavelength_item = QtWidgets.QTableWidgetItem(
                f'{ds.wavelength_nm:.1f}' if ds.wavelength_nm else 'N/A'
            )
            self.datasets_table.setItem(row, 1, wavelength_item)

            power_item = QtWidgets.QTableWidgetItem(
                f'{ds.laser_power_mW:.3g}' if ds.laser_power_mW else 'N/A'
            )
            self.datasets_table.setItem(row, 2, power_item)

            port_power_item = QtWidgets.QTableWidgetItem(
                f'{ds.port_power_mW:.3g}' if ds.port_power_mW else 'N/A'
            )
            self.datasets_table.setItem(row, 3, port_power_item)

            pixel_size_item = QtWidgets.QTableWidgetItem(
                f'{ds.pixel_size_um:.2f}' if ds.pixel_size_um else 'N/A'
            )
            self.datasets_table.setItem(row, 4, pixel_size_item)

    def clear_plots(self):
        for widget in self._plot_widgets:
            widget.close()
        self._plot_widgets = []

    def update_plots(self, plot_widgets: list[pg.GraphicsLayoutWidget]):
        self.tab_widget.clear()
        for widget in plot_widgets:
            self.tab_widget.addTab(widget, widget.windowTitle())

    def run_analysis(self):
        if not self._datasets:
            logger.info('No Datasets | Please add at least one dataset to analyze.')
            return

        if not self.sphere_calibration_tool._cal:
            logger.info(
                'No Sphere Calibration | Please perform sphere calibration first.'
            )
            return

        self._set_controls_enabled(False)
        self.progress_bar.setValue(0)

        def analysis_finished():
            self.progress_bar.setValue(100)
            logger.info('Analysis complete.')

            self._set_controls_enabled(True)

        def update_progress(value):
            if isinstance(value, str):
                logger.info(value)
            elif isinstance(value, (int, float)):
                self.progress_bar.setValue(int(value))

        def done_callback(result):
            if result is None:
                return

            exposure_times_s, returned_data = result
            self._exposure_times_s = np.array(exposure_times_s)
            self._returned_data = returned_data
            self._sync_dataset_calibration_fields_from_results()

            self._plot_widgets = plot_results(
                self._datasets,
                self._returned_data,
                self._exposure_times_s,
            )
            self._results_table = results_to_dataframe(
                self._datasets,
                self._returned_data,
            )
            self.update_plots(self._plot_widgets)
            self.show_results_table()

        worker = QThreadWorker(
            perform_analysis,
            self._datasets,
            self.sphere_calibration_tool,
            variance_source=self._selected_gain_variance_source(),
            progress=True,
        )
        worker.signals.progress.connect(update_progress)
        worker.signals.result.connect(done_callback)
        worker.signals.finished.connect(analysis_finished)

        QtCore.QThreadPool.globalInstance().start(worker)

    def save_analysis_cache(self):
        if (
            not self._datasets
            or self._returned_data is None
            or self._exposure_times_s is None
        ):
            logger.info('No analysis cache to save. Run analysis first.')
            return

        file_path, _ = getSaveFileName(
            self,
            'Save Photon Transfer Analysis Cache',
            '',
            'NumPy Files (*.npz)',
        )
        if not file_path:
            return

        if not file_path.lower().endswith('.npz'):
            file_path = f'{file_path}.npz'

        try:
            count = save_analysis_cache(
                file_path,
                self._datasets,
                self._returned_data,
                self._exposure_times_s,
                gain_variance_source=self._selected_gain_variance_source(),
            )
            logger.info(f'Saved analysis cache for {count} dataset(s) to {file_path}')
        except Exception as e:
            logger.error(f'Failed to save analysis cache: {e}')

    def load_analysis_cache(self):
        file_path, _ = getOpenFileName(
            self,
            'Load Photon Transfer Analysis Cache',
            '',
            'NumPy Files (*.npz)',
        )
        if not file_path:
            return

        try:
            (
                datasets,
                returned_data,
                exposure_times_s,
                analysis_options,
            ) = load_analysis_cache(file_path)
        except Exception as e:
            logger.error(f'Failed to load analysis cache: {e}')
            return

        if not datasets or not returned_data:
            logger.info('Loaded cache is empty.')
            return

        self._datasets = datasets
        self._returned_data = returned_data
        self._exposure_times_s = exposure_times_s
        self._sync_dataset_calibration_fields_from_results()

        loaded_variance_source = analysis_options.get('gain_variance_source')
        if loaded_variance_source is None:
            for params in self._returned_data.values():
                if (
                    isinstance(params, dict)
                    and params.get('fit_variance_source') is not None
                ):
                    loaded_variance_source = str(params.get('fit_variance_source'))
                    break
        self._set_gain_variance_source(loaded_variance_source)

        self.update_directory_list()

        self._plot_widgets = plot_results(
            self._datasets,
            self._returned_data,
            self._exposure_times_s,
        )
        self._results_table = results_to_dataframe(
            self._datasets,
            self._returned_data,
        )
        self.update_plots(self._plot_widgets)
        self.progress_bar.setValue(100)
        self.show_results_table()

        logger.info(
            f'Loaded analysis cache with {len(self._datasets)} dataset(s) '
            f'from {file_path} '
            f'(gain variance source: {self._selected_gain_variance_source()})'
        )

    def export_dark_calibration_json(self):
        if not self._datasets:
            logger.info('No datasets available to export.')
            return

        if self._returned_data is None:
            logger.info(
                'No cached analysis results available. '
                'Run analysis or load analysis cache first.'
            )
            return

        file_path, _ = getSaveFileName(
            self,
            'Export Dark Calibration Mapping JSON',
            '',
            'JSON (*.json)',
        )
        if not file_path:
            return

        if not file_path.lower().endswith('.json'):
            file_path = f'{file_path}.json'

        payload = {}
        for name, ds in self._datasets.items():
            params = self._returned_data.get(name, {})
            gain = to_float_or_none(
                params.get('gain_e_per_dn') if isinstance(params, dict) else None
            )
            if gain is None:
                gain = to_float_or_none(ds.gain)

            responsivity = to_float_or_none(
                params.get('responsivity') if isinstance(params, dict) else None
            )
            if responsivity is None:
                responsivity = to_float_or_none(ds.responsivity)

            quantum_efficiency = to_float_or_none(
                params.get('qe') if isinstance(params, dict) else None
            )
            if quantum_efficiency is None:
                quantum_efficiency = to_float_or_none(ds.quantum_efficiency)

            payload[name] = {
                'dark_calibration_directory': '',
                'gain': gain,
                'responsivity': responsivity,
                'quantum_efficiency': quantum_efficiency,
            }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            logger.info(
                f'Exported dark calibration mapping JSON for '
                f'{len(payload)} dataset(s) to {file_path}'
            )
        except Exception as e:
            logger.error(f'Failed to export dark calibration mapping JSON: {e}')

    def compare_caches(self):
        left_path, _ = getOpenFileName(
            self,
            'Select Cache A',
            '',
            'NumPy Files (*.npz)',
        )
        if not left_path:
            return

        right_path, _ = getOpenFileName(
            self,
            'Select Cache B',
            os.path.dirname(left_path),
            'NumPy Files (*.npz)',
        )
        if not right_path:
            return

        try:
            (
                _left_datasets,
                left_data,
                _left_exposures,
                left_options,
            ) = load_analysis_cache(left_path)
        except Exception as e:
            logger.error(f'Failed to load cache A: {e}')
            return

        try:
            (
                _right_datasets,
                right_data,
                _right_exposures,
                right_options,
            ) = load_analysis_cache(right_path)
        except Exception as e:
            logger.error(f'Failed to load cache B: {e}')
            return

        dialog = CacheComparisonDialog(
            left_label=os.path.basename(left_path),
            right_label=os.path.basename(right_path),
            left_data=left_data,
            right_data=right_data,
            left_options=left_options,
            right_options=right_options,
            parent=self,
        )
        dialog.show()

    def show_results_table(self):
        if self._results_table is None:
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('PTC Analysis Results')
        layout = QtWidgets.QVBoxLayout(dialog)

        table_view = QtWidgets.QTableView()
        model = PandasModel(self._results_table)
        table_view.setModel(model)
        table_view.resizeColumnsToContents()

        layout.addWidget(table_view)

        export_button = QtWidgets.QPushButton('Export to HDF')
        layout.addWidget(export_button)
        export_button.clicked.connect(self.export_results)

        dialog.resize(800, 400)
        dialog.show()

    def export_results(self):
        if self._results_table is None:
            return

        h5_path, _ = getSaveFileName(self, 'Export Results to HDF', '', 'HDF5 (*.h5)')
        if not h5_path:
            return

        try:
            self._results_table.to_hdf(h5_path, key='results', mode='w')
            logger.info(f'Results exported to {h5_path}')
        except Exception as e:
            logger.error(f'Failed to export results: {e}')


if __name__ == '__main__':
    import sys

    from microEye.qt import QApplication

    app = QApplication(sys.argv)

    reg_widget = PhotonTransfer()
    reg_widget.show()

    sys.exit(app.exec())
