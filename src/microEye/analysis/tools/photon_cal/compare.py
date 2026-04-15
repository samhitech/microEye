import logging
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg

from microEye.analysis.tools.photon_cal.table import PandasModel
from microEye.qt import Qt, QtWidgets

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

        (
            self._comparison_df,
            self._common_names,
            self._left_only,
            self._right_only,
        ) = build_scalar_comparison(self._left_data, self._right_data)

        self._setup_ui()
        self._update_scalar_plots()
        self._update_curve_overlay()

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
        layout = QtWidgets.QVBoxLayout(self)

        label_controls = QtWidgets.QHBoxLayout()
        layout.addLayout(label_controls)
        label_controls.addWidget(QtWidgets.QLabel('A label:'))
        self.a_label_edit = QtWidgets.QLineEdit('A')
        self.a_label_edit.setToolTip('Legend label for cache A')
        label_controls.addWidget(self.a_label_edit)
        label_controls.addSpacing(12)
        label_controls.addWidget(QtWidgets.QLabel('B label:'))
        self.b_label_edit = QtWidgets.QLineEdit('B')
        self.b_label_edit.setToolTip('Legend label for cache B')
        label_controls.addWidget(self.b_label_edit)
        label_controls.addStretch(1)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        summary_widget = QtWidgets.QTextEdit()
        summary_widget.setReadOnly(True)
        summary_widget.setPlainText(self._build_summary_text())
        tabs.addTab(summary_widget, 'Summary')

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
        self.metric_bars_plot.plotItem.setTitle(
            'Metric by Dataset (Blue=A, Orange=B)'
        )
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

        self.overlay_plot = pg.PlotWidget()
        self.overlay_plot.setWindowTitle('Curve Overlay')
        self.overlay_plot.plotItem.setTitle('Curve Overlay')
        self.overlay_plot.plotItem.addLegend()
        self.overlay_plot.plotItem.showGrid(x=True, y=True)
        curves_layout.addWidget(self.overlay_plot, 1)

        tabs.addTab(curves_widget, 'Dataset Curves')

        self.metric_combo.currentTextChanged.connect(self._update_scalar_plots)
        self.delta_mode_combo.currentTextChanged.connect(self._update_scalar_plots)
        self.dataset_combo.currentTextChanged.connect(self._update_curve_overlay)
        self.curve_combo.currentTextChanged.connect(self._update_curve_overlay)
        self.a_label_edit.textChanged.connect(self._update_scalar_plots)
        self.b_label_edit.textChanged.connect(self._update_scalar_plots)
        self.a_label_edit.textChanged.connect(self._update_curve_overlay)
        self.b_label_edit.textChanged.connect(self._update_curve_overlay)

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
        lines.append('Median absolute percent deltas:')
        grouped = self._comparison_df.groupby('Metric', dropna=False)
        for metric, frame in grouped:
            values = np.abs(frame['Delta %'].to_numpy(dtype=float))
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                lines.append(f'  {metric}: n/a')
            else:
                lines.append(f'  {metric}: {np.median(finite):.3g}%')

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
        self.metric_bars_plot.plotItem.setTitle(
            f'{metric_label}: {a_label} and {b_label} by dataset'
        )
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
            self.delta_plot.plotItem.setTitle(
                f'{metric_label}: percent delta by dataset'
            )
            self.delta_plot.plotItem.setLabel(
                'left', f'Delta % ({b_label}-{a_label})/{a_label}'
            )
        else:
            self.delta_plot.plotItem.setTitle(
                f'{metric_label}: absolute delta by dataset'
            )
            self.delta_plot.plotItem.setLabel('left', f'Delta ({b_label}-{a_label})')

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

    def _update_curve_overlay(self):
        a_label, b_label = self._comparison_labels()
        dataset_name = self.dataset_combo.currentText()
        curve_kind = self.curve_combo.currentText()

        self._clear_and_ensure_legend(self.overlay_plot)

        if not dataset_name:
            self.overlay_plot.plotItem.setTitle(
                'Curve Overlay: no common dataset selected'
            )
            return

        left_params = self._left_data.get(dataset_name, {})
        right_params = self._right_data.get(dataset_name, {})

        if curve_kind == 'Gain Curve':
            left_curve = _gain_curve(left_params)
            right_curve = _gain_curve(right_params)
            self.overlay_plot.plotItem.setTitle(f'Gain curve overlay: {dataset_name}')
            self.overlay_plot.plotItem.setLabel('bottom', 'Mean Signal [ADU]')
            self.overlay_plot.plotItem.setLabel('left', 'Variance [ADU^2]')
        elif curve_kind == 'QE Curve':
            left_curve = _qe_curve(left_params)
            right_curve = _qe_curve(right_params)
            self.overlay_plot.plotItem.setTitle(f'QE curve overlay: {dataset_name}')
            self.overlay_plot.plotItem.setLabel(
                'bottom', 'Photon Flux [photons/cm^2/s]'
            )
            self.overlay_plot.plotItem.setLabel('left', 'Mean Signal [e-]')
        else:
            left_curve = _snr_curve(left_params)
            right_curve = _snr_curve(right_params)
            self.overlay_plot.plotItem.setTitle(f'SNR curve overlay: {dataset_name}')
            self.overlay_plot.plotItem.setLabel(
                'bottom', 'Photon Flux [photons/cm^2/s]'
            )
            self.overlay_plot.plotItem.setLabel('left', 'SNR [dB]')

        if left_curve is not None:
            self.overlay_plot.plot(
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
            self.overlay_plot.plot(
                right_curve[0],
                right_curve[1],
                pen=pg.mkPen((230, 145, 45), width=2),
                name=f'{b_label}: {self._right_label}',
            )
        else:
            logger.warning(
                f'No plottable {curve_kind} data in B for dataset {dataset_name}.'
            )
