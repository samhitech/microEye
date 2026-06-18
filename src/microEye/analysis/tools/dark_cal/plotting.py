# ruff: noqa: I001

import logging
import os
from typing import Callable

import numpy as np
import pyqtgraph as pg

from sklearn.linear_model import LinearRegression

from microEye.analysis.tools.dark_cal.constants import (
    HISTOGRAM_DATA_TYPES,
    DataTypes,
)
from microEye.utils.pyqt2mplt import (
    MatplotlibPlotterDialog,
    SubplotPayload,
    FigurePayload,
    PlotSeriesPayload,
)

logger = logging.getLogger(__name__)


def plot_and_fit(x: np.ndarray, y: np.ndarray, plot_item: pg.PlotItem, **kwargs):

    model = LinearRegression().fit(x.reshape(-1, 1), y)
    r2 = model.score(x.reshape(-1, 1), y)

    plot_item.plot(
        x,
        y,
        pen=None,
        symbol='o',
        symbolSize=6,
        symbolPen=kwargs.get('symbolPen', 'b'),
        symbolBrush=kwargs.get('symbolBrush', 'b'),
        name='Data Points',
    )
    plot_item.plot(
        x,
        model.predict(x.reshape(-1, 1)),
        pen=pg.mkPen(kwargs.get('lineColor', 'r'), width=2),
        name=f'Linear Fit (R²={r2:.3f}) | '
        f'y={model.coef_[0]:.3f}x+{model.intercept_:.3f}',
    )


def plot_and_fit_payload(
    x: np.ndarray, y: np.ndarray, **kwargs
) -> list[PlotSeriesPayload]:
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    r2 = model.score(x.reshape(-1, 1), y)

    series = [
        PlotSeriesPayload(
            x=x,
            y=y,
            plot_type='scatter',
            style={
                'marker': 'o',
                'edgecolor': kwargs.get('symbolPen', 'b'),
                'facecolor': kwargs.get('symbolBrush', 'b'),
            },
        ),
        PlotSeriesPayload(
            x=x,
            y=model.predict(x.reshape(-1, 1)),
            plot_type='line',
            style={
                'color': kwargs.get('lineColor', 'r'),
                'linestyle': '--',
            },
            label=f'Linear Fit (R²={r2:.3f}) | '
            f'y={model.coef_[0]:.3f}x+{model.intercept_:.3f}',
        ),
    ]

    return series


def plot_dark_cal(
    plot_window: pg.GraphicsLayoutWidget, data: dict[DataTypes, np.ndarray]
):
    exposure_plot: pg.PlotItem = plot_window.addPlot(row=1, col=0)
    exposure_plot.setTitle('Mean Intensity vs Exposure Time')
    exposure_plot.setLabel('left', 'Mean Intensity [ADU]')
    exposure_plot.setLabel('bottom', 'Exposure Time (s)')

    variance_plot: pg.PlotItem = plot_window.addPlot(row=1, col=1)
    variance_plot.setTitle('Variance vs Exposure Time')
    variance_plot.setLabel('left', 'Variance [ADU²]')
    variance_plot.setLabel('bottom', 'Exposure Time (s)')

    gain_plot: pg.PlotItem = plot_window.addPlot(row=2, col=0)
    gain_plot.setTitle('Variance vs Mean Intensity')
    gain_plot.setLabel('left', 'Variance [ADU²]')
    gain_plot.setLabel('bottom', 'Mean Intensity [ADU]')

    temperature_plot: pg.PlotItem = plot_window.addPlot(row=2, col=1)
    temperature_plot.setTitle('Temperature vs Exposure Time')
    temperature_plot.setLabel('left', 'Temperature [°C]')
    temperature_plot.setLabel('bottom', 'Exposure Time (s)')

    exposure_plot.addLegend()
    variance_plot.addLegend()
    gain_plot.addLegend()
    temperature_plot.addLegend()

    exposure_times = data[DataTypes.EXPOSURE] / 1000.0
    mean = data[DataTypes.MEAN]
    variance = data[DataTypes.VARIANCE]
    temp = data.get(DataTypes.TEMPERATURE)

    plot_and_fit(
        exposure_times,
        mean,
        exposure_plot,
        symbolPen='b',
        symbolBrush='b',
        lineColor='r',
    )
    plot_and_fit(
        exposure_times,
        variance,
        variance_plot,
        symbolPen='g',
        symbolBrush='g',
        lineColor='m',
    )
    plot_and_fit(
        mean,
        variance,
        gain_plot,
        symbolPen='y',
        symbolBrush='y',
        lineColor='c',
    )

    if temp is not None:
        temperature_plot.plot(
            exposure_times,
            temp,
            pen=None,
            symbol='o',
            symbolSize=6,
            symbolPen='orange',
            symbolBrush='orange',
            name='Data Points',
        )


def plot_histograms(
    plot_window: pg.GraphicsLayoutWidget, data: dict[DataTypes, dict], gain: float
):
    convert_funcs = {
        DataTypes.BASELINE: lambda G, DATA: DATA * G,
        DataTypes.DARK_CURRENT: lambda G, DATA: DATA * G,
        DataTypes.DARK_VARIANCE: lambda G, DATA: DATA * (G**2),
        DataTypes.THERMAL_VARIANCE: lambda G, DATA: DATA * (G**2),
        DataTypes.DARK_NOISE: lambda G, DATA: DATA * G,
        DataTypes.THERMAL_NOISE: lambda G, DATA: DATA * G,
    }
    for i, data_type in enumerate(HISTOGRAM_DATA_TYPES):
        hist_data = data[data_type]
        converter: Callable = convert_funcs.get(data_type, lambda G, DATA: DATA)
        median = converter(gain, hist_data['median'])

        plot_item = plot_window.addPlot(row=(i // 2) + 1, col=i % 2)
        plot_item.setTitle(data_type.value.replace('_', ' ').title())
        plot_item.setLabel('left', 'Count')
        plot_item.setLabel(
            'bottom',
            f"{data_type.value.replace('_', ' ').title()}"
            f" (Centered at Median {median:.3f})",
        )
        plot_item.plot(
            converter(gain, hist_data['bin_edges']),
            hist_data['hist'],
            stepMode=True,
            fillLevel=0,
            brush=(100, 100, 255, 150),
        )


def plot_results_pyqtgraph(
    directory: str, data: dict, meta: dict, mode: str
) -> pg.GraphicsLayoutWidget:
    try:
        parent_dir = os.path.basename(os.path.dirname(directory))
        plot_window = pg.GraphicsLayoutWidget(title=f'{os.path.basename(directory)}')

        label = pg.LabelItem(
            f'{parent_dir} - {os.path.basename(directory)}', size='16pt', bold=True
        )
        plot_window.addItem(label, row=0, col=0, colspan=2)

        gain = meta.get(directory, {}).get('gain', 1.0)

        if mode == 'Histograms':
            plot_histograms(plot_window, data, gain)
        else:
            plot_dark_cal(plot_window, data)

        return plot_window
    except Exception as e:
        logger.error(f'Error plotting results for {directory}: {e}')
        return None


def plot_results(
    directories: dict,
    meta: dict,
    mode: str,
    matplotlib_plotter: MatplotlibPlotterDialog = None,
) -> list[pg.GraphicsLayoutWidget]:
    plot_widgets = []

    meta = meta or {}

    for _, (directory, data) in enumerate(directories.items()):
        try:
            plot_window = plot_results_pyqtgraph(directory, data, meta, mode)
            if plot_window is not None:
                plot_widgets.append(plot_window)
        except Exception as e:
            logger.error(f'Error plotting results for {directory}: {e}')

    return plot_widgets


def get_histogram_payload(
    idx: int,
    directory: str,
    data: dict,
    meta: dict,
    **kwargs,
) -> SubplotPayload:
    gain = meta.get(directory, {}).get('gain', 1.0)

    convert_funcs = {
        DataTypes.BASELINE: lambda G, DATA: DATA * G,
        DataTypes.DARK_CURRENT: lambda G, DATA: DATA * G,
        DataTypes.DARK_VARIANCE: lambda G, DATA: DATA * (G**2),
        DataTypes.THERMAL_VARIANCE: lambda G, DATA: DATA * (G**2),
        DataTypes.DARK_NOISE: lambda G, DATA: DATA * G,
        DataTypes.THERMAL_NOISE: lambda G, DATA: DATA * G,
    }

    subplots = []

    for i, type in enumerate(HISTOGRAM_DATA_TYPES):
        converter: Callable = convert_funcs.get(type, lambda G, DATA: DATA)
        median = converter(gain, data[type]['median'])
        hist_data = data[type]

        edges = converter(gain, hist_data['bin_edges'])

        bin_centers = (edges[:-1] + edges[1:]) / 2

        hist_sum = np.nansum(hist_data['hist'])
        normalized_hist = (
            hist_data['hist'] / hist_sum if hist_sum != 0 else hist_data['hist']
        )

        series_list = [
            PlotSeriesPayload(
                x=bin_centers,
                y=normalized_hist,
                plot_type='fill',
                label=os.path.basename(directory),
                dataset=os.path.basename(directory),
                style={
                    'facecolor': '#6464ff',
                    'alpha': 0.6,
                },
            )
        ]

        # convert idx to letter: 0 -> A, 1 -> B, etc.
        title = f'{chr(65 + idx)}) ' if len(meta) > 1 else ''
        title = meta.get(directory, {}).get('name', title)

        xlabel = (
            f'{type.to_symbol()} ({median:.1f} {type.to_unit(gain)})'
            # f" ({median:.1f} $\\pm$ {data[type]['mad']:.2f} {type.to_unit(gain)})"
        )

        subplots.append(
            SubplotPayload(
                row=i,
                col=idx,
                title=title if i == 0 else '',
                xlabel=xlabel,
                ylabel='Count',
                series=series_list,
                metadata={
                    'title_alignment': 'center',
                    'title_bold': True,
                    'legend': False,
                },
            )
        )

    return subplots


def get_dark_cal_payload(
    idx: int, directory: str, data: dict, meta: dict
) -> list[SubplotPayload]:

    gain = meta.get(directory, {}).get('gain', 1.0)

    exposure_times = data[DataTypes.EXPOSURE] / 1000.0
    mean = data[DataTypes.MEAN] * gain
    variance = data[DataTypes.VARIANCE] * (gain**2)
    temp = data.get(DataTypes.TEMPERATURE)

    mean_series = plot_and_fit_payload(
        exposure_times, mean, symbolPen='b', symbolBrush='b', lineColor='r'
    )
    variance_series = plot_and_fit_payload(
        exposure_times, variance, symbolPen='g', symbolBrush='g', lineColor='m'
    )
    gain_series = plot_and_fit_payload(
        mean, variance, symbolPen='y', symbolBrush='y', lineColor='c'
    )
    temp_series = [
        PlotSeriesPayload(
            x=exposure_times,
            y=temp,
            plot_type='scatter',
            style={
                'marker': 'o',
                'edgecolor': 'orange',
                'facecolor': 'orange',
            },
            label='Temperature',
        )
    ]

    # title is 'A)', 'B)', etc. based on idx, but only if there are multiple datasets
    # convert idx to letter: 0 -> A, 1 -> B, etc.
    title = f'{chr(65 + idx)}) ' if len(meta) > 1 else ''
    title = meta.get(directory, {}).get('name', title)

    return [
        SubplotPayload(
            row=0,
            col=idx,
            title=title,
            xlabel='Exposure Time [s]',
            ylabel=f'Mean Intensity [{DataTypes.MEAN.to_unit(gain)}]',
            series=mean_series,
            metadata={
                'title_alignment': 'center',
                'title_bold': True,
                'legend': False,
            },
        ),
        SubplotPayload(
            row=1,
            col=idx,
            title='',
            xlabel='Exposure Time [s]',
            ylabel=f'Variance [{DataTypes.VARIANCE.to_unit(gain)}]',
            series=variance_series,
            metadata={
                'title_alignment': 'left',
                'title_bold': True,
                'legend': False,
            },
        ),
        SubplotPayload(
            row=2,
            col=idx,
            title='',
            xlabel=f'Mean Intensity [{DataTypes.MEAN.to_unit(gain)}]',
            ylabel=f'Variance [{DataTypes.VARIANCE.to_unit(gain)}]',
            series=gain_series,
            metadata={
                'title_alignment': 'left',
                'title_bold': True,
                'legend': False,
            },
        ),
        SubplotPayload(
            row=3,
            col=idx,
            title='',
            xlabel='Exposure Time [s]',
            ylabel=f'Temperature [°C]',
            series=temp_series,
            metadata={
                'title_alignment': 'left',
                'title_bold': True,
                'legend': True,
            },
        ),
    ]


DEFAULT_HIST_COLS = 4
DEFAULT_DARK_CAL_ROWS = 4


def plot_results_matplotlib(
    directories: dict,
    dataset_meta: dict,
    mode: str,
    **kwargs,
) -> MatplotlibPlotterDialog:
    plotter = MatplotlibPlotterDialog()

    subplots = []

    dataset_meta = dataset_meta or {}

    for idx, (directory, data) in enumerate(directories.items()):
        try:
            if mode == 'Histograms':
                subplots.extend(
                    get_histogram_payload(idx, directory, data, dataset_meta, **kwargs)
                )
            else:
                subplots.extend(
                    get_dark_cal_payload(idx, directory, data, dataset_meta)
                )
        except Exception as e:
            logger.error(f'Error plotting results for {directory}: {e}')

    title = 'Histograms' if mode == 'Histograms' else 'Dark Calibration Results'

    rows = DEFAULT_HIST_COLS if mode == 'Histograms' else DEFAULT_DARK_CAL_ROWS
    cols = len(directories)

    fig = FigurePayload(
        title=title,
        subplots=subplots,
        metadata={
            'layout': {
                'rows': rows,
                'cols': cols,
                'width_in': cols * 3,
                'height_in': rows * 3,
            },
        },
    )

    plotter.set_payload(fig)

    return plotter
