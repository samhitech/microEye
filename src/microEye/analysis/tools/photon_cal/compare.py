import logging
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg

from microEye.analysis.tools.photon_cal.table import PandasModel
from microEye.qt import Qt, QtWidgets
from microEye.utils.pyqt2mplt import (
    FigurePayload,
    MatplotlibPlotterDialog,
    PlotSeriesPayload,
    SubplotPayload,
)

logger = logging.getLogger(__name__)


SCALAR_METRICS = [
    ('Gain [e-/DN]', 'gain_e_per_dn', 1.0),
    ('Read Noise [e-]', 'read_noise_e', 1.0),
    ('QE [%]', 'qe', 100.0),
    ('Responsivity [ADU/photon]', 'responsivity', 1.0),
]

CURVE_TYPES = [
    'Gain Curve',
    'QE Curve',
    'SNR Curve',
]

MPL_PLOT_SPECS = (
    ('parity', 'Parity scatter'),
    ('bars', 'Metric bars'),
    ('delta', 'Delta bars'),
    ('comparison', 'Curve comparison'),
)


class PlotSelectionDialog(QtWidgets.QDialog):
    def __init__(self, selected_keys: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select Matplotlib Plots')

        self._checkboxes: dict[str, QtWidgets.QCheckBox] = {}

        layout = QtWidgets.QFormLayout(self)
        layout.addRow(
            QtWidgets.QLabel('Choose the comparison plots that should be exported:')
        )

        for key, label in MPL_PLOT_SPECS:
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(key in selected_keys)
            checkbox.setToolTip(f'Include {label.lower()} in the Matplotlib view')
            self._checkboxes[key] = checkbox
            layout.addWidget(checkbox)

        button_row = QtWidgets.QHBoxLayout()
        layout.addRow(button_row)

        self.select_all_button = QtWidgets.QPushButton('All')
        self.select_none_button = QtWidgets.QPushButton('None')
        self.select_all_button.clicked.connect(self._select_all)
        self.select_none_button.clicked.connect(self._select_none)
        button_row.addWidget(self.select_all_button)
        button_row.addWidget(self.select_none_button)
        button_row.addStretch(1)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    def _select_all(self):
        for checkbox in self._checkboxes.values():
            checkbox.setChecked(True)

    def _select_none(self):
        for checkbox in self._checkboxes.values():
            checkbox.setChecked(False)

    def selected_keys(self) -> list[str]:
        return [
            key for key, checkbox in self._checkboxes.items() if checkbox.isChecked()
        ]


def _as_scalar(value: Any) -> float | None:
    arr = np.asarray(value)
    if arr.ndim != 0:
        return None
    scalar = arr.item()
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    try:
        return float(scalar)
    except Exception:
        return None


def _safe_percent_delta(a: float, b: float) -> float:
    if not np.isfinite(a) or abs(a) < 1e-12:
        return np.nan
    return (b - a) / a * 100.0


def build_scalar_comparison(
    left_data: dict[str, dict],
    right_data: dict[str, dict],
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    left_names = set(left_data.keys())
    right_names = set(right_data.keys())

    common = sorted(left_names.intersection(right_names))
    left_only = sorted(left_names.difference(right_names))
    right_only = sorted(right_names.difference(left_names))

    rows = []
    for name in common:
        left_params = left_data.get(name, {})
        right_params = right_data.get(name, {})

        for metric_label, metric_key, scale in SCALAR_METRICS:
            left_scalar = _as_scalar(left_params.get(metric_key))
            right_scalar = _as_scalar(right_params.get(metric_key))

            if left_scalar is None or right_scalar is None:
                continue

            a = left_scalar * scale
            b = right_scalar * scale
            rows.append(
                {
                    'Dataset': name,
                    'Metric': metric_label,
                    'A': a,
                    'B': b,
                    'Delta Abs (B-A)': b - a,
                    'Delta %': _safe_percent_delta(a, b),
                }
            )

    return pd.DataFrame.from_records(rows), common, left_only, right_only


def _metric_by_label(label: str) -> tuple[str, float]:
    for metric_label, metric_key, scale in SCALAR_METRICS:
        if metric_label == label:
            return metric_key, scale
    return SCALAR_METRICS[0][1], SCALAR_METRICS[0][2]


def _gain_curve(params: dict) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(params.get('mean_dn', []))
    y = np.asarray(params.get('fit_variance_dn2', params.get('var_shot_dn2', [])))
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return None
    return x[mask], y[mask]


def _qe_curve(params: dict) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(params.get('photons_per_pixel', []))
    mean_e = np.asarray(params.get('mean_e', []))
    if x.ndim != 1 or mean_e.ndim != 1 or len(mean_e) < 2:
        return None
    y = mean_e[1:]
    if len(x) != len(y):
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return None
    return x[mask], y[mask]


def _snr_curve(params: dict) -> tuple[np.ndarray, np.ndarray] | None:
    photons = np.asarray(params.get('photons_per_pixel', []))
    mean_e = np.asarray(params.get('mean_e', []))
    var_total = np.asarray(params.get('var_total_dn2', []))
    gain = _as_scalar(params.get('gain_e_per_dn'))

    if photons.ndim != 1 or mean_e.ndim != 1 or var_total.ndim != 1:
        return None
    if len(mean_e) < 2 or len(var_total) < 2 or gain is None:
        return None

    var_e = var_total[1:] * (gain**2)
    signal = mean_e[1:]

    if len(photons) != len(signal) or len(signal) != len(var_e):
        return None

    mask = (
        (var_e > 0)
        & np.isfinite(var_e)
        & np.isfinite(signal)
        & np.isfinite(photons)
        & (signal > 0)
    )
    if not np.any(mask):
        return None

    snr = signal[mask] / np.sqrt(var_e[mask])
    snr_db = 20.0 * np.log10(snr)
    return photons[mask], snr_db


class CacheComparisonDialog(QtWidgets.QDialog):
    def __init__(
        self,
        left_label: str,
        right_label: str,
        left_data: dict[str, dict],
        right_data: dict[str, dict],
        left_options: dict | None,
        right_options: dict | None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle('PTC Cache Comparison')
        self.resize(1300, 900)

        self._left_label = left_label
        self._right_label = right_label
        self._left_data = left_data
        self._right_data = right_data
        self._left_options = left_options or {}
        self._right_options = right_options or {}
        self._mpl_plotter_dialog: MatplotlibPlotterDialog | None = None
        self._mpl_selected_plot_keys: list[str] = [
            key for key, _label in MPL_PLOT_SPECS
        ]

        (
            self._comparison_df,
            self._common_names,
            self._left_only,
            self._right_only,
        ) = build_scalar_comparison(self._left_data, self._right_data)

        self._setup_ui()
        self._update_scalar_plots()
        self._update_curve_comparison()

    def _comparison_labels(self) -> tuple[str, str]:
        a_label = (
            self.a_label_edit.text().strip() if hasattr(self, 'a_label_edit') else ''
        )
        b_label = (
            self.b_label_edit.text().strip() if hasattr(self, 'b_label_edit') else ''
        )
        return (a_label or 'A', b_label or 'B')

    @staticmethod
    def _clear_and_ensure_legend(plot_widget: pg.PlotWidget):
        plot_item = plot_widget.plotItem
        plot_widget.clear()
        if plot_item.legend is None:
            plot_item.addLegend()

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        compare_box = QtWidgets.QGroupBox('Comparison Settings')
        form_layout = QtWidgets.QFormLayout(compare_box)

        layout.addWidget(compare_box, 2)

        self.a_label_edit = QtWidgets.QLineEdit('A')
        self.a_label_edit.setToolTip('Legend label for cache A')
        form_layout.addRow(QtWidgets.QLabel('A label:'), self.a_label_edit)

        self.b_label_edit = QtWidgets.QLineEdit('B')
        self.b_label_edit.setToolTip('Legend label for cache B')
        form_layout.addRow(QtWidgets.QLabel('B label:'), self.b_label_edit)

        self.choose_mpl_plots_button = QtWidgets.QPushButton('Select Plots...')
        self.choose_mpl_plots_button.setToolTip(
            'Pick which comparison plots are sent to the Matplotlib dialog'
        )
        self.choose_mpl_plots_button.clicked.connect(self._choose_matplotlib_plots)

        self.open_mpl_plotter_button = QtWidgets.QPushButton('Open Plotter')
        self.open_mpl_plotter_button.setToolTip('Open the Matplotlib dialog')
        self.open_mpl_plotter_button.clicked.connect(self._open_matplotlib_plotter)

        self.export_hdf_table_button = QtWidgets.QPushButton('Export Table')
        self.export_hdf_table_button.setToolTip(
            'Export the comparison table to an HDF file'
        )
        self.export_hdf_table_button.clicked.connect(self._export_hdf_table)

        form_layout.addWidget(self.choose_mpl_plots_button)
        form_layout.addWidget(self.open_mpl_plotter_button)
        form_layout.addWidget(self.export_hdf_table_button)

        form_layout.addRow(QtWidgets.QLabel('Summary:'))

        summary_widget = QtWidgets.QTextEdit()
        summary_widget.setReadOnly(True)
        summary_widget.setPlainText(self._build_summary_text())
        form_layout.addRow(summary_widget)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 4)

        table_widget = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_widget)

        table_view = QtWidgets.QTableView()
        table_model = PandasModel(self._comparison_df)
        table_view.setModel(table_model)
        table_view.resizeColumnsToContents()
        table_layout.addWidget(table_view)

        tabs.addTab(table_widget, 'Table')

        scalar_widget = QtWidgets.QWidget()
        scalar_layout = QtWidgets.QVBoxLayout(scalar_widget)

        scalar_controls = QtWidgets.QHBoxLayout()
        scalar_layout.addLayout(scalar_controls)

        scalar_controls.addWidget(QtWidgets.QLabel('Metric:'))
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems([m[0] for m in SCALAR_METRICS])
        scalar_controls.addWidget(self.metric_combo)

        scalar_controls.addSpacing(20)
        scalar_controls.addWidget(QtWidgets.QLabel('Difference:'))
        self.delta_mode_combo = QtWidgets.QComboBox()
        self.delta_mode_combo.addItems(['Percent (%)', 'Absolute (B-A)'])
        scalar_controls.addWidget(self.delta_mode_combo)
        scalar_controls.addStretch(1)

        self.parity_plot = pg.PlotWidget()
        self.parity_plot.setWindowTitle('Metric Parity')
        self.parity_plot.plotItem.setTitle('Metric Parity (A vs B)')
        self.parity_plot.plotItem.setLabel('bottom', self._left_label)
        self.parity_plot.plotItem.setLabel('left', self._right_label)
        self.parity_plot.plotItem.addLegend()
        self.parity_plot.plotItem.showGrid(x=True, y=True)
        scalar_layout.addWidget(self.parity_plot, 1)

        self.metric_bars_plot = pg.PlotWidget()
        self.metric_bars_plot.setWindowTitle('Metric Bars')
        self.metric_bars_plot.plotItem.setTitle('Metric by Dataset (Blue=A, Orange=B)')
        self.metric_bars_plot.plotItem.setLabel('bottom', 'Dataset')
        self.metric_bars_plot.plotItem.addLegend()
        self.metric_bars_plot.plotItem.showGrid(x=True, y=True)
        scalar_layout.addWidget(self.metric_bars_plot, 1)

        self.delta_plot = pg.PlotWidget()
        self.delta_plot.setWindowTitle('Metric Difference')
        self.delta_plot.plotItem.setTitle('Difference by Dataset')
        self.delta_plot.plotItem.setLabel('bottom', 'Dataset')
        self.delta_plot.plotItem.addLegend()
        self.delta_plot.plotItem.showGrid(x=True, y=True)
        scalar_layout.addWidget(self.delta_plot, 1)

        tabs.addTab(scalar_widget, 'Scalar Plots')

        curves_widget = QtWidgets.QWidget()
        curves_layout = QtWidgets.QVBoxLayout(curves_widget)

        curve_controls = QtWidgets.QHBoxLayout()
        curves_layout.addLayout(curve_controls)

        curve_controls.addWidget(QtWidgets.QLabel('Dataset:'))
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.addItems(self._common_names)
        curve_controls.addWidget(self.dataset_combo)

        curve_controls.addSpacing(20)
        curve_controls.addWidget(QtWidgets.QLabel('Curve:'))
        self.curve_combo = QtWidgets.QComboBox()
        self.curve_combo.addItems(CURVE_TYPES)
        curve_controls.addWidget(self.curve_combo)
        curve_controls.addStretch(1)

        self.compare_plot = pg.PlotWidget()
        self.compare_plot.setWindowTitle('Curve Comparison')
        self.compare_plot.plotItem.setTitle('Curve Comparison')
        self.compare_plot.plotItem.addLegend()
        self.compare_plot.plotItem.showGrid(x=True, y=True)
        curves_layout.addWidget(self.compare_plot, 1)

        tabs.addTab(curves_widget, 'Dataset Curves')

        self.metric_combo.currentTextChanged.connect(self._update_scalar_plots)
        self.delta_mode_combo.currentTextChanged.connect(self._update_scalar_plots)
        self.dataset_combo.currentTextChanged.connect(self._update_curve_comparison)
        self.curve_combo.currentTextChanged.connect(self._update_curve_comparison)
        self.a_label_edit.textChanged.connect(self._update_scalar_plots)
        self.b_label_edit.textChanged.connect(self._update_scalar_plots)
        self.a_label_edit.textChanged.connect(self._update_curve_comparison)
        self.b_label_edit.textChanged.connect(self._update_curve_comparison)

    def _build_summary_text(self) -> str:
        left_source = self._left_options.get('gain_variance_source', 'unknown')
        right_source = self._right_options.get('gain_variance_source', 'unknown')

        lines = [
            f'Cache A: {self._left_label}',
            f'Cache B: {self._right_label}',
            '',
            f'Common datasets: {len(self._common_names)}',
            f'Only in A: {len(self._left_only)}',
            f'Only in B: {len(self._right_only)}',
            '',
            f'Gain variance source A: {left_source}',
            f'Gain variance source B: {right_source}',
        ]

        if self._left_only:
            lines.extend(['', 'Datasets only in A: ' + ', '.join(self._left_only)])
        if self._right_only:
            lines.extend(['', 'Datasets only in B: ' + ', '.join(self._right_only)])

        if self._comparison_df.empty:
            lines.extend(['', 'No comparable scalar metrics were found.'])
            return '\n'.join(lines)

        lines.append('')

        return '\n'.join(lines)

    def _collect_scalar_series(
        self,
        metric_key: str,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        names: list[str] = []
        a_vals: list[float] = []
        b_vals: list[float] = []
        abs_delta_vals: list[float] = []
        pct_delta_vals: list[float] = []

        for name in self._common_names:
            left_params = self._left_data.get(name, {})
            right_params = self._right_data.get(name, {})

            left_scalar = _as_scalar(left_params.get(metric_key))
            right_scalar = _as_scalar(right_params.get(metric_key))
            if left_scalar is None or right_scalar is None:
                continue

            a = left_scalar * scale
            b = right_scalar * scale
            if not (np.isfinite(a) and np.isfinite(b)):
                continue

            names.append(name)
            a_vals.append(a)
            b_vals.append(b)
            abs_delta_vals.append(b - a)
            pct_delta_vals.append(_safe_percent_delta(a, b))

        return (
            np.asarray(a_vals, dtype=float),
            np.asarray(b_vals, dtype=float),
            np.asarray(abs_delta_vals, dtype=float),
            np.asarray(pct_delta_vals, dtype=float),
            names,
        )

    def _update_scalar_plots(self):
        a_label, b_label = self._comparison_labels()
        metric_label = self.metric_combo.currentText()
        metric_key, scale = _metric_by_label(metric_label)

        a_vals, b_vals, abs_delta, pct_delta, names = self._collect_scalar_series(
            metric_key,
            scale,
        )

        self._clear_and_ensure_legend(self.parity_plot)
        self.parity_plot.plotItem.setTitle(
            f'{metric_label}: parity ({a_label} vs {b_label})'
        )
        self.parity_plot.plotItem.setLabel('bottom', a_label)
        self.parity_plot.plotItem.setLabel('left', b_label)

        if len(a_vals) > 0:
            self.parity_plot.plot(
                a_vals,
                b_vals,
                pen=None,
                symbol='o',
                symbolBrush=pg.mkBrush(45, 125, 210),
                symbolPen=None,
                name=f'{a_label} vs {b_label}',
            )
            lo = float(np.nanmin(np.concatenate([a_vals, b_vals])))
            hi = float(np.nanmax(np.concatenate([a_vals, b_vals])))
            if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                self.parity_plot.plot(
                    [lo, hi],
                    [lo, hi],
                    pen=pg.mkPen((180, 180, 180), style=Qt.PenStyle.DashLine),
                    name='Parity line',
                )

        self._clear_and_ensure_legend(self.metric_bars_plot)
        self.metric_bars_plot.plotItem.setTitle(f'{metric_label}')
        self.metric_bars_plot.plotItem.setLabel('left', metric_label)

        if len(a_vals) > 0:
            idx = np.arange(len(a_vals), dtype=float)
            width = 0.36
            offset = width / 2.0

            bars_a = pg.BarGraphItem(
                x=idx - offset,
                height=a_vals,
                width=width,
                brush=pg.mkBrush(45, 125, 210),
                pen=pg.mkPen(35, 95, 160),
            )
            bars_b = pg.BarGraphItem(
                x=idx + offset,
                height=b_vals,
                width=width,
                brush=pg.mkBrush(230, 145, 45),
                pen=pg.mkPen(190, 115, 30),
            )
            self.metric_bars_plot.addItem(bars_a)
            self.metric_bars_plot.addItem(bars_b)

            self.metric_bars_plot.plot(
                [np.nan],
                [np.nan],
                pen=pg.mkPen(35, 95, 160, width=3),
                name=a_label,
            )
            self.metric_bars_plot.plot(
                [np.nan],
                [np.nan],
                pen=pg.mkPen(190, 115, 30, width=3),
                name=b_label,
            )

            self.metric_bars_plot.plotItem.getAxis('bottom').setTicks(
                [list(zip(idx, names))]
            )

        self._clear_and_ensure_legend(self.delta_plot)
        show_percent = self.delta_mode_combo.currentIndex() == 0
        delta_values = pct_delta if show_percent else abs_delta

        if show_percent:
            self.delta_plot.plotItem.setTitle(f'{metric_label}: percent delta')
            self.delta_plot.plotItem.setLabel('left', 'Δ [%]')
        else:
            self.delta_plot.plotItem.setTitle(f'{metric_label}: absolute delta')
            self.delta_plot.plotItem.setLabel('left', f'Δ ({b_label}-{a_label})')

        if len(delta_values) > 0:
            idx = np.arange(len(delta_values), dtype=float)
            delta_clean = np.nan_to_num(delta_values, nan=0.0)
            bar = pg.BarGraphItem(
                x=idx,
                height=delta_clean,
                width=0.75,
                brushes=[
                    pg.mkBrush(220, 80, 80) if v > 0 else pg.mkBrush(70, 160, 90)
                    for v in delta_clean
                ],
            )
            self.delta_plot.addItem(bar)
            self.delta_plot.plot(
                [np.nan],
                [np.nan],
                pen=pg.mkPen((160, 160, 160), width=2),
                name=f'{b_label} - {a_label}',
            )
            self.delta_plot.plotItem.getAxis('bottom').setTicks([list(zip(idx, names))])
            self.delta_plot.plot(
                [float(idx.min()) - 0.6, float(idx.max()) + 0.6],
                [0.0, 0.0],
                pen=pg.mkPen((160, 160, 160), style=Qt.PenStyle.DashLine),
            )

        self._update_mpl_plotter_payload()

    def _update_curve_comparison(self):
        a_label, b_label = self._comparison_labels()
        dataset_name = self.dataset_combo.currentText()
        curve_kind = self.curve_combo.currentText()

        self._clear_and_ensure_legend(self.compare_plot)

        if not dataset_name:
            self.compare_plot.plotItem.setTitle(
                'Curve Overlay: no common dataset selected'
            )
            return

        left_params = self._left_data.get(dataset_name, {})
        right_params = self._right_data.get(dataset_name, {})

        if curve_kind == 'Gain Curve':
            left_curve = _gain_curve(left_params)
            right_curve = _gain_curve(right_params)
            self.compare_plot.plotItem.setTitle(f'Gain curve: {dataset_name}')
            self.compare_plot.plotItem.setLabel('bottom', 'Mean Signal [ADU]')
            self.compare_plot.plotItem.setLabel('left', 'Variance [ADU^2]')
        elif curve_kind == 'QE Curve':
            left_curve = _qe_curve(left_params)
            right_curve = _qe_curve(right_params)
            self.compare_plot.plotItem.setTitle(f'QE curve: {dataset_name}')
            self.compare_plot.plotItem.setLabel(
                'bottom', 'Photon Flux [photons/cm^2/s]'
            )
            self.compare_plot.plotItem.setLabel('left', 'Mean Signal [e-]')
        else:
            left_curve = _snr_curve(left_params)
            right_curve = _snr_curve(right_params)
            self.compare_plot.plotItem.setTitle(f'SNR curve: {dataset_name}')
            self.compare_plot.plotItem.setLabel(
                'bottom', 'Photon Flux [photons/cm^2/s]'
            )
            self.compare_plot.plotItem.setLabel('left', 'SNR [dB]')

        if left_curve is not None:
            self.compare_plot.plot(
                left_curve[0],
                left_curve[1],
                pen=pg.mkPen((45, 125, 210), width=2),
                name=f'{a_label}: {self._left_label}',
            )
        else:
            logger.warning(
                f'No plottable {curve_kind} data in A for dataset {dataset_name}.'
            )

        if right_curve is not None:
            self.compare_plot.plot(
                right_curve[0],
                right_curve[1],
                pen=pg.mkPen((230, 145, 45), width=2),
                name=f'{b_label}: {self._right_label}',
            )
        else:
            logger.warning(
                f'No plottable {curve_kind} data in B for dataset {dataset_name}.'
            )

        self._update_mpl_plotter_payload()

    def _export_hdf_table(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Export Comparison Table',
            filter='HDF5 Files (*.h5);;All Files (*)',
        )
        if not path:
            return

        try:
            self._comparison_df.to_hdf(path, key='comparison', mode='w')
            QtWidgets.QMessageBox.information(
                self,
                'Export Successful',
                f'Comparison table successfully exported to:\n{path}',
            )
        except Exception as e:
            logger.exception('Failed to export comparison table to HDF5')
            QtWidgets.QMessageBox.critical(
                self,
                'Export Failed',
                f'An error occurred while exporting the table:\n{str(e)}',
            )

    def _open_matplotlib_plotter(self):
        if self._mpl_plotter_dialog is None:
            self._mpl_plotter_dialog = MatplotlibPlotterDialog(self)
            self._mpl_plotter_dialog.destroyed.connect(
                lambda *_: setattr(self, '_mpl_plotter_dialog', None)
            )

        self._mpl_plotter_dialog.show()
        self._mpl_plotter_dialog.raise_()
        self._mpl_plotter_dialog.activateWindow()
        self._update_mpl_plotter_payload()

    def _choose_matplotlib_plots(self):
        dialog = PlotSelectionDialog(self._mpl_selected_plot_keys, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        selected_keys = dialog.selected_keys()
        if not selected_keys:
            selected_keys = [key for key, _label in MPL_PLOT_SPECS]

        self._mpl_selected_plot_keys = selected_keys
        self._update_mpl_plotter_payload()

    def _update_mpl_plotter_payload(self):
        if self._mpl_plotter_dialog is None:
            return

        payload = self._build_mpl_payload()
        self._mpl_plotter_dialog.set_payload(payload)

    def _selected_plot_specs(self) -> list[tuple[str, str]]:
        selected = set(self._mpl_selected_plot_keys)
        return [spec for spec in MPL_PLOT_SPECS if spec[0] in selected]

    @staticmethod
    def _plot_position(index: int, layout_cols: int) -> tuple[int, int]:
        return index // layout_cols, index % layout_cols

    def _build_parity_subplot(
        self,
        *,
        row: int,
        col: int,
        metric_label: str,
        a_label: str,
        b_label: str,
        a_vals: np.ndarray,
        b_vals: np.ndarray,
    ) -> SubplotPayload:
        series: list[PlotSeriesPayload] = []
        if len(a_vals) > 0:
            series.append(
                PlotSeriesPayload(
                    x=a_vals,
                    y=b_vals,
                    plot_type='scatter',
                    label=f'{a_label} vs {b_label}',
                    dataset='Parity',
                    style={'color': '#2d7dd2', 'alpha': 0.9},
                )
            )

            lo = float(np.nanmin(np.concatenate([a_vals, b_vals])))
            hi = float(np.nanmax(np.concatenate([a_vals, b_vals])))
            if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                series.append(
                    PlotSeriesPayload(
                        x=np.asarray([lo, hi], dtype=float),
                        y=np.asarray([lo, hi], dtype=float),
                        plot_type='line',
                        label='Parity line',
                        dataset='Reference',
                        style={
                            'color': '#9c9c9c',
                            'linestyle': '--',
                            'linewidth': 1.4,
                        },
                    )
                )

        return self._subplot_payload(
            row=row,
            col=col,
            title=f'{metric_label}: parity ({a_label} vs {b_label})',
            xlabel=a_label,
            ylabel=b_label,
            series=series,
        )

    def _build_bar_subplot(
        self,
        *,
        row: int,
        col: int,
        metric_label: str,
        a_label: str,
        b_label: str,
        a_vals: np.ndarray,
        b_vals: np.ndarray,
        names: list[str],
    ) -> SubplotPayload:
        series: list[PlotSeriesPayload] = []
        if len(a_vals) > 0:
            idx = np.arange(len(a_vals), dtype=float)
            width = 0.36
            offset = width / 2.0
            series.extend(
                [
                    PlotSeriesPayload(
                        x=idx - offset,
                        y=a_vals,
                        plot_type='bar',
                        label=a_label,
                        dataset=a_label,
                        style={
                            'width': width,
                            'color': '#2d7dd2',
                            'edgecolor': '#235fa0',
                        },
                        extras={'width': width},
                    ),
                    PlotSeriesPayload(
                        x=idx + offset,
                        y=b_vals,
                        plot_type='bar',
                        label=b_label,
                        dataset=b_label,
                        style={
                            'width': width,
                            'color': '#e6912d',
                            'edgecolor': '#be731e',
                        },
                        extras={'width': width},
                    ),
                ]
            )

        return self._subplot_payload(
            row=row,
            col=col,
            title=f'{metric_label}: {a_label} and {b_label}',
            xlabel='Dataset Index',
            ylabel=metric_label,
            series=series,
            metadata={'dataset_names': names},
        )

    def _build_delta_subplot(
        self,
        *,
        row: int,
        col: int,
        metric_label: str,
        a_label: str,
        b_label: str,
        delta_values: np.ndarray,
        show_percent: bool,
        names: list[str],
    ) -> SubplotPayload:
        series: list[PlotSeriesPayload] = []
        if len(delta_values) > 0:
            idx = np.arange(len(delta_values), dtype=float)
            delta_clean = np.nan_to_num(delta_values, nan=0.0)
            series.append(
                PlotSeriesPayload(
                    x=idx,
                    y=delta_clean,
                    plot_type='bar',
                    label=f'{b_label} - {a_label}',
                    dataset='Δ',
                    style={
                        'width': 0.75,
                        'color': [
                            '#dc5050' if value > 0 else '#46a15a'
                            for value in delta_clean
                        ],
                    },
                    extras={'width': 0.75},
                )
            )
            series.append(
                PlotSeriesPayload(
                    x=np.asarray([float(idx.min()) - 0.6, float(idx.max()) + 0.6]),
                    y=np.asarray([0.0, 0.0]),
                    plot_type='line',
                    label='Baseline',
                    dataset='Reference',
                    style={
                        'color': '#9c9c9c',
                        'linestyle': '--',
                        'linewidth': 1.4,
                    },
                )
            )

        delta_ylabel = 'Δ [%]' if show_percent else f'Δ ({b_label}-{a_label})'
        delta_title = (
            f'{metric_label}: percent delta'
            if show_percent
            else f'{metric_label}: absolute delta'
        )
        return self._subplot_payload(
            row=row,
            col=col,
            title=delta_title,
            xlabel='Dataset Index',
            ylabel=delta_ylabel,
            series=series,
            metadata={'dataset_names': names},
        )

    def _build_comparison_subplot(
        self,
        *,
        row: int,
        col: int,
        dataset_name: str,
        curve_kind: str,
        a_label: str,
        b_label: str,
    ) -> SubplotPayload:
        series: list[PlotSeriesPayload] = []
        if not dataset_name:
            return self._subplot_payload(
                row=row,
                col=col,
                title='Curve Overlay: no common dataset selected',
                xlabel='',
                ylabel='',
                series=series,
            )

        left_params = self._left_data.get(dataset_name, {})
        right_params = self._right_data.get(dataset_name, {})

        if curve_kind == 'Gain Curve':
            left_curve = _gain_curve(left_params)
            right_curve = _gain_curve(right_params)
            title = f'Gain curve: {dataset_name}'
            xlabel = 'Mean Signal [ADU]'
            ylabel = 'Variance [ADU^2]'
        elif curve_kind == 'QE Curve':
            left_curve = _qe_curve(left_params)
            right_curve = _qe_curve(right_params)
            title = f'QE curve: {dataset_name}'
            xlabel = 'Photon Flux [photons/cm^2/s]'
            ylabel = 'Mean Signal [e-]'
        else:
            left_curve = _snr_curve(left_params)
            right_curve = _snr_curve(right_params)
            title = f'SNR curve: {dataset_name}'
            xlabel = 'Photon Flux [photons/cm^2/s]'
            ylabel = 'SNR [dB]'

        if left_curve is not None:
            series.append(
                PlotSeriesPayload(
                    x=left_curve[0],
                    y=left_curve[1],
                    plot_type='line',
                    label=f'{a_label}: {self._left_label}',
                    dataset=a_label,
                    style={'color': '#2d7dd2', 'linewidth': 2.0},
                )
            )
        if right_curve is not None:
            series.append(
                PlotSeriesPayload(
                    x=right_curve[0],
                    y=right_curve[1],
                    plot_type='line',
                    label=f'{b_label}: {self._right_label}',
                    dataset=b_label,
                    style={'color': '#e6912d', 'linewidth': 2.0},
                )
            )

        return self._subplot_payload(
            row=row,
            col=col,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            series=series,
        )

    def _build_mpl_payload(self) -> FigurePayload:
        a_label, b_label = self._comparison_labels()
        metric_label = self.metric_combo.currentText()
        metric_key, scale = _metric_by_label(metric_label)
        show_percent = self.delta_mode_combo.currentIndex() == 0
        curve_kind = self.curve_combo.currentText()
        dataset_name = self.dataset_combo.currentText()
        selected_plot_specs = [
            spec for spec in MPL_PLOT_SPECS if spec[0] in self._mpl_selected_plot_keys
        ]
        layout_rows, layout_cols = self._preferred_layout(len(selected_plot_specs))

        a_vals, b_vals, abs_delta, pct_delta, names = self._collect_scalar_series(
            metric_key,
            scale,
        )

        subplots: list[SubplotPayload] = []
        for subplot_index, (key, _label) in enumerate(selected_plot_specs):
            row, col = divmod(subplot_index, layout_cols)

            if key == 'parity':
                series: list[PlotSeriesPayload] = []
                if len(a_vals) > 0:
                    series.append(
                        PlotSeriesPayload(
                            x=a_vals,
                            y=b_vals,
                            plot_type='scatter',
                            label=f'{a_label} vs {b_label}',
                            dataset='Parity',
                            style={'color': '#2d7dd2', 'alpha': 0.9},
                        )
                    )

                    lo = float(np.nanmin(np.concatenate([a_vals, b_vals])))
                    hi = float(np.nanmax(np.concatenate([a_vals, b_vals])))
                    if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                        series.append(
                            PlotSeriesPayload(
                                x=np.asarray([lo, hi], dtype=float),
                                y=np.asarray([lo, hi], dtype=float),
                                plot_type='line',
                                label='Parity line',
                                dataset='Reference',
                                style={
                                    'color': '#9c9c9c',
                                    'linestyle': '--',
                                    'linewidth': 1.4,
                                },
                            )
                        )

                subplots.append(
                    SubplotPayload(
                        row=row,
                        col=col,
                        title=f'{metric_label}: parity ({a_label} vs {b_label})',
                        xlabel=a_label,
                        ylabel=b_label,
                        series=series,
                    )
                )
                continue

            if key == 'bars':
                series = []
                if len(a_vals) > 0:
                    idx = np.arange(len(a_vals), dtype=float)
                    width = 0.36
                    offset = width / 2.0
                    series.extend(
                        [
                            PlotSeriesPayload(
                                x=idx - offset,
                                y=a_vals,
                                plot_type='bar',
                                label=a_label,
                                dataset=a_label,
                                style={
                                    'width': width,
                                    'color': '#2d7dd2',
                                    'edgecolor': '#235fa0',
                                },
                                extras={'width': width},
                            ),
                            PlotSeriesPayload(
                                x=idx + offset,
                                y=b_vals,
                                plot_type='bar',
                                label=b_label,
                                dataset=b_label,
                                style={
                                    'width': width,
                                    'color': '#e6912d',
                                    'edgecolor': '#be731e',
                                },
                                extras={'width': width},
                            ),
                        ]
                    )

                subplots.append(
                    SubplotPayload(
                        row=row,
                        col=col,
                        title=f'{metric_label}: {a_label} and {b_label}',
                        xlabel='Dataset Index',
                        ylabel=metric_label,
                        series=series,
                        metadata={'dataset_names': names},
                    )
                )
                continue

            if key == 'delta':
                delta_values = pct_delta if show_percent else abs_delta
                series = []
                if len(delta_values) > 0:
                    idx = np.arange(len(delta_values), dtype=float)
                    delta_clean = np.nan_to_num(delta_values, nan=0.0)
                    series.append(
                        PlotSeriesPayload(
                            x=idx,
                            y=delta_clean,
                            plot_type='bar',
                            label=f'{b_label} - {a_label}',
                            dataset='Δ',
                            style={
                                'width': 0.75,
                                'color': [
                                    '#dc5050' if value > 0 else '#46a15a'
                                    for value in delta_clean
                                ],
                            },
                            extras={'width': 0.75},
                        )
                    )
                    series.append(
                        PlotSeriesPayload(
                            x=np.asarray(
                                [float(idx.min()) - 0.6, float(idx.max()) + 0.6]
                            ),
                            y=np.asarray([0.0, 0.0]),
                            plot_type='line',
                            label='Baseline',
                            dataset='Reference',
                            style={
                                'color': '#9c9c9c',
                                'linestyle': '--',
                                'linewidth': 1.4,
                            },
                        )
                    )

                delta_ylabel = 'Δ [%]' if show_percent else f'Δ ({b_label}-{a_label})'
                delta_title = (
                    f'{metric_label}: percent delta'
                    if show_percent
                    else f'{metric_label}: absolute delta'
                )
                subplots.append(
                    SubplotPayload(
                        row=row,
                        col=col,
                        title=delta_title,
                        xlabel='Dataset Index',
                        ylabel=delta_ylabel,
                        series=series,
                        metadata={'dataset_names': names},
                    )
                )
                continue

            if key == 'comparison':
                series = []
                if dataset_name:
                    left_params = self._left_data.get(dataset_name, {})
                    right_params = self._right_data.get(dataset_name, {})

                    if curve_kind == 'Gain Curve':
                        left_curve = _gain_curve(left_params)
                        right_curve = _gain_curve(right_params)
                        title = f'Gain curve: {dataset_name}'
                        xlabel = 'Mean Signal [ADU]'
                        ylabel = 'Variance [ADU^2]'
                    elif curve_kind == 'QE Curve':
                        left_curve = _qe_curve(left_params)
                        right_curve = _qe_curve(right_params)
                        title = f'QE curve: {dataset_name}'
                        xlabel = 'Photon Flux [photons/cm^2/s]'
                        ylabel = 'Mean Signal [e-]'
                    else:
                        left_curve = _snr_curve(left_params)
                        right_curve = _snr_curve(right_params)
                        title = f'SNR curve: {dataset_name}'
                        xlabel = 'Photon Flux [photons/cm^2/s]'
                        ylabel = 'SNR [dB]'

                    if left_curve is not None:
                        series.append(
                            PlotSeriesPayload(
                                x=left_curve[0],
                                y=left_curve[1],
                                plot_type='line',
                                label=f'{a_label}: {self._left_label}',
                                dataset=a_label,
                                style={'color': '#2d7dd2', 'linewidth': 2.0},
                            )
                        )
                    if right_curve is not None:
                        series.append(
                            PlotSeriesPayload(
                                x=right_curve[0],
                                y=right_curve[1],
                                plot_type='line',
                                label=f'{b_label}: {self._right_label}',
                                dataset=b_label,
                                style={'color': '#e6912d', 'linewidth': 2.0},
                            )
                        )
                else:
                    title = 'Curve Overlay: no common dataset selected'
                    xlabel = ''
                    ylabel = ''

                subplots.append(
                    SubplotPayload(
                        row=row,
                        col=col,
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        series=series,
                    )
                )

        return FigurePayload(
            title=f'PTC Cache Comparison ({a_label} vs {b_label})',
            subplots=subplots,
            metadata={
                'layout': {
                    'rows': layout_rows,
                    'cols': layout_cols,
                }
            },
        )

    @staticmethod
    def _preferred_layout(count: int) -> tuple[int, int]:
        if count <= 0:
            return 1, 1

        cols = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / cols))
        return rows, cols

    @staticmethod
    def _subplot_payload(
        *,
        row: int,
        col: int,
        title: str,
        xlabel: str,
        ylabel: str,
        series: list[PlotSeriesPayload],
        metadata: dict[str, Any] | None = None,
    ) -> SubplotPayload:
        return SubplotPayload(
            row=row,
            col=col,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            series=series,
            metadata=metadata or {},
        )
