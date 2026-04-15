import logging

import numpy as np
import pandas as pd
import pyqtgraph as pg

from microEye.analysis.tools.common import normalize_positive_float, to_bool
from microEye.analysis.tools.dark_cal.constants import HISTOGRAM_DATA_TYPES, DataTypes
from microEye.qt import Qt, QtCore, QtWidgets

logger = logging.getLogger(__name__)


METRICS = [
    ('Baseline [e-]', 'baseline_e'),
    ('Dark Current [e-/s]', 'dark_current_e_per_s'),
    ('Noise Intercept [e-^2]', 'noise_intercept_e2'),
    ('Noise Slope [e-^2/s]', 'noise_slope_e2_per_s'),
]


def _safe_percent_delta(reference: float, value: float) -> float:
    if not np.isfinite(reference) or abs(reference) < 1e-12:
        return np.nan
    return (value - reference) / reference * 100.0


def _resolve_gain(metadata: dict) -> tuple[float, bool]:
    gain, gain_defaulted = normalize_positive_float(metadata.get('gain'), default=1.0)
    if not gain_defaulted:
        gain_defaulted = to_bool(metadata.get('gain_defaulted'), default=False)
    return gain, gain_defaulted


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return None

    x_valid = x[mask]
    y_valid = y[mask]

    design = np.vstack([np.ones_like(x_valid), x_valid]).T
    coeff, *_ = np.linalg.lstsq(design, y_valid, rcond=None)
    intercept, slope = coeff
    return float(intercept), float(slope)


def _standard_metrics(data: dict) -> dict[str, float] | None:
    exposure = np.asarray(data.get(DataTypes.EXPOSURE, []), dtype=np.float64)
    mean = np.asarray(data.get(DataTypes.MEAN, []), dtype=np.float64)
    variance = np.asarray(data.get(DataTypes.VARIANCE, []), dtype=np.float64)

    if exposure.ndim != 1 or mean.ndim != 1 or variance.ndim != 1:
        return None
    if len(exposure) != len(mean) or len(exposure) != len(variance):
        return None

    exposure_s = exposure / 1000.0

    mean_fit = _fit_line(exposure_s, mean)
    var_fit = _fit_line(exposure_s, variance)
    if mean_fit is None or var_fit is None:
        return None

    baseline, dark_current = mean_fit
    noise_intercept, noise_slope = var_fit

    return {
        'baseline_adu': baseline,
        'dark_current_adu_per_s': dark_current,
        'noise_intercept_adu2': noise_intercept,
        'noise_slope_adu2_per_s': noise_slope,
    }


def _histogram_metrics(data: dict) -> dict[str, float] | None:
    if not all(key in data for key in HISTOGRAM_DATA_TYPES):
        return None

    try:
        return {
            'baseline_adu': float(data[DataTypes.BASELINE]['median']),
            'dark_current_adu_per_s': float(data[DataTypes.DARK_CURRENT]['median']),
            'noise_intercept_adu2': float(data[DataTypes.DARK_NOISE]['median']),
            'noise_slope_adu2_per_s': float(data[DataTypes.THERMAL_NOISE]['median']),
        }
    except Exception:
        return None


def _extract_metrics(mode: str, data: dict) -> dict[str, float] | None:
    if mode == 'Histograms':
        return _histogram_metrics(data)
    return _standard_metrics(data)


def _build_display_labels(
    directories: list[str],
    names: dict[str, str],
) -> dict[str, str]:
    labels: dict[str, str] = {}
    used: dict[str, int] = {}

    for directory in directories:
        base = str(names.get(directory, '')).strip()
        if not base:
            base = directory.replace('\\', '/').rstrip('/').split('/')[-1] or directory

        count = used.get(base, 0) + 1
        used[base] = count
        labels[directory] = base if count == 1 else f'{base}_{count}'

    return labels


def build_comparison_dataframe(
    directories: dict,
    mode: str,
    dataset_meta: dict[str, dict] | None = None,
) -> pd.DataFrame:
    rows = []
    defaulted_count = 0
    meta_lookup = dataset_meta or {}

    names = {
        directory: str(meta_lookup.get(directory, {}).get('name', ''))
        for directory in directories
    }
    labels = _build_display_labels(list(directories.keys()), names)

    for directory, data in directories.items():
        metrics = _extract_metrics(mode, data)
        if metrics is None:
            logger.warning(f'Skipping {directory}: insufficient data for comparison.')
            continue

        metadata = meta_lookup.get(directory, {})
        gain, gain_defaulted = _resolve_gain(metadata)
        if gain_defaulted:
            defaulted_count += 1

        rows.append(
            {
                'Dataset': labels[directory],
                'Source Path': directory,
                'Gain [e-/ADU]': gain,
                'Baseline [ADU]': metrics['baseline_adu'],
                'Dark Current [ADU/s]': metrics['dark_current_adu_per_s'],
                'Noise Intercept [ADU^2]': metrics['noise_intercept_adu2'],
                'Noise Slope [ADU^2/s]': metrics['noise_slope_adu2_per_s'],
                'Baseline [e-]': metrics['baseline_adu'] * gain,
                'Dark Current [e-/s]': metrics['dark_current_adu_per_s'] * gain,
                'Noise Intercept [e-^2]': metrics['noise_intercept_adu2'] * (gain**2),
                'Noise Slope [e-^2/s]': metrics['noise_slope_adu2_per_s'] * (gain**2),
            }
        )

    frame = pd.DataFrame.from_records(rows)
    frame.attrs['gain_defaulted_count'] = defaulted_count
    return frame


class _PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, frame: pd.DataFrame):
        super().__init__()
        self._frame = frame

    def rowCount(self, parent=None):
        return self._frame.shape[0]

    def columnCount(self, parent=None):
        return self._frame.shape[1]

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            value = self._frame.iat[index.row(), index.column()]
            if isinstance(value, (float, np.floating)):
                return f'{value:.6g}'
            return str(value)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return self._frame.columns[section]
        return str(section)


class DarkDatasetComparisonDialog(QtWidgets.QDialog):
    def __init__(
        self,
        directories: dict,
        mode: str,
        dataset_meta: dict[str, dict] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle('Dark Calibration Dataset Comparison')
        self.resize(1200, 850)

        self._mode = mode
        self._frame = build_comparison_dataframe(directories, mode, dataset_meta)

        self._setup_ui()
        self._update_plots()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        summary = QtWidgets.QTextEdit()
        summary.setReadOnly(True)
        summary.setPlainText(self._build_summary_text())
        tabs.addTab(summary, 'Summary')

        table_tab = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_tab)
        table_view = QtWidgets.QTableView()
        table_view.setModel(_PandasModel(self._frame))
        table_view.resizeColumnsToContents()
        table_layout.addWidget(table_view)
        tabs.addTab(table_tab, 'Table')

        plots_tab = QtWidgets.QWidget()
        plots_layout = QtWidgets.QVBoxLayout(plots_tab)

        controls = QtWidgets.QHBoxLayout()
        plots_layout.addLayout(controls)

        controls.addWidget(QtWidgets.QLabel('Metric:'))
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems([label for label, _key in METRICS])
        controls.addWidget(self.metric_combo)

        controls.addSpacing(16)
        controls.addWidget(QtWidgets.QLabel('Reference Dataset:'))
        self.reference_combo = QtWidgets.QComboBox()
        if not self._frame.empty:
            self.reference_combo.addItems(self._frame['Dataset'].astype(str).tolist())
        controls.addWidget(self.reference_combo)

        controls.addSpacing(16)
        controls.addWidget(QtWidgets.QLabel('Difference:'))
        self.difference_combo = QtWidgets.QComboBox()
        self.difference_combo.addItems(['Percent (%)', 'Absolute'])
        controls.addWidget(self.difference_combo)
        controls.addStretch(1)

        self.value_plot = pg.PlotWidget()
        self.value_plot.setWindowTitle('Metric Values')
        self.value_plot.plotItem.showGrid(x=True, y=True)
        plots_layout.addWidget(self.value_plot, 2)

        self.delta_plot = pg.PlotWidget()
        self.delta_plot.setWindowTitle('Difference To Reference')
        self.delta_plot.plotItem.showGrid(x=True, y=True)
        plots_layout.addWidget(self.delta_plot, 2)

        tabs.addTab(plots_tab, 'Plots')

        self.metric_combo.currentTextChanged.connect(self._update_plots)
        self.reference_combo.currentTextChanged.connect(self._update_plots)
        self.difference_combo.currentTextChanged.connect(self._update_plots)

    def _build_summary_text(self) -> str:
        lines = [
            f'Mode: {self._mode}',
            f'Datasets compared: {len(self._frame)}',
            '',
            'Metrics:',
            '- Baseline [e-] (mean-vs-exposure intercept converted by gain)',
            '- Dark Current [e-/s] (mean-vs-exposure slope converted by gain)',
            '- Noise Intercept [e-^2] (variance intercept converted by gain^2)',
            '- Noise Slope [e-^2/s] (variance slope converted by gain^2)',
        ]

        if self._mode == 'Histograms':
            lines.extend(
                [
                    '',
                    'Histogram mode uses map medians for comparison values.',
                ]
            )

        if self._frame.empty:
            lines.extend(['', 'No comparable datasets are available.'])

        defaulted_count = int(self._frame.attrs.get('gain_defaulted_count', 0))
        if defaulted_count > 0:
            lines.extend(
                [
                    '',
                    (
                        f'Warning: {defaulted_count} dataset(s) '
                        'are using default gain=1.0.'
                    ),
                    (
                        'Converted electron metrics for those datasets '
                        'may be meaningless.'
                    ),
                ]
            )

        return '\n'.join(lines)

    def _metric_column(self) -> str:
        selected = self.metric_combo.currentText()
        for label, _key in METRICS:
            if label == selected:
                return label
        return METRICS[0][0]

    def _update_plots(self):
        metric_column = self._metric_column()

        self.value_plot.clear()
        self.delta_plot.clear()

        if self._frame.empty:
            self.value_plot.plotItem.setTitle('No data to plot')
            self.delta_plot.plotItem.setTitle('No data to plot')
            return

        labels = self._frame['Dataset'].astype(str).tolist()
        values = self._frame[metric_column].to_numpy(dtype=float)
        idx = np.arange(len(values), dtype=float)

        self.value_plot.plotItem.setTitle(f'{metric_column} by dataset')
        self.value_plot.plotItem.setLabel('bottom', 'Dataset')
        self.value_plot.plotItem.setLabel('left', metric_column)

        value_bars = pg.BarGraphItem(
            x=idx,
            height=values,
            width=0.7,
            brush=pg.mkBrush(70, 135, 220),
            pen=pg.mkPen(45, 95, 160),
        )
        self.value_plot.addItem(value_bars)
        self.value_plot.plotItem.getAxis('bottom').setTicks([list(zip(idx, labels))])

        reference_name = self.reference_combo.currentText()
        if reference_name and reference_name in labels:
            reference_idx = labels.index(reference_name)
        else:
            reference_idx = 0

        reference_value = values[reference_idx]
        use_percent = self.difference_combo.currentIndex() == 0

        if use_percent:
            delta = np.array(
                [_safe_percent_delta(reference_value, val) for val in values],
                dtype=float,
            )
            delta_label = 'Delta %'
            self.delta_plot.plotItem.setTitle(
                f'{metric_column}: percent difference to {labels[reference_idx]}'
            )
        else:
            delta = values - reference_value
            delta_label = 'Delta (absolute)'
            self.delta_plot.plotItem.setTitle(
                f'{metric_column}: absolute difference to {labels[reference_idx]}'
            )

        delta_clean = np.nan_to_num(delta, nan=0.0)
        delta_bars = pg.BarGraphItem(
            x=idx,
            height=delta_clean,
            width=0.7,
            brushes=[
                pg.mkBrush(220, 90, 90) if v > 0 else pg.mkBrush(80, 170, 95)
                for v in delta_clean
            ],
        )
        self.delta_plot.addItem(delta_bars)
        self.delta_plot.plotItem.setLabel('bottom', 'Dataset')
        self.delta_plot.plotItem.setLabel('left', delta_label)
        self.delta_plot.plotItem.getAxis('bottom').setTicks([list(zip(idx, labels))])

        self.delta_plot.plot(
            [float(idx.min()) - 0.6, float(idx.max()) + 0.6],
            [0.0, 0.0],
            pen=pg.mkPen((160, 160, 160), style=Qt.PenStyle.DashLine),
        )
