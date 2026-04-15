# ruff: noqa: I001

import logging
import os

import numpy as np
import pyqtgraph as pg

from microEye.analysis.tools.dark_cal.constants import (
    HISTOGRAM_DATA_TYPES,
    DataTypes,
)

logger = logging.getLogger(__name__)


def plot_and_fit(x, y, plot_item: pg.PlotItem, **kwargs):
    from sklearn.linear_model import LinearRegression

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


def plot_results(directories: dict, mode: str) -> list[pg.GraphicsLayoutWidget]:
    plot_widgets = []

    for _, (directory, data) in enumerate(directories.items()):
        try:
            parent_dir = os.path.basename(os.path.dirname(directory))
            plot_window = pg.GraphicsLayoutWidget(
                title=f'{os.path.basename(directory)}'
            )

            label = pg.LabelItem(
                f'{parent_dir} - {os.path.basename(directory)}', size='16pt', bold=True
            )
            plot_window.addItem(label, row=0, col=0, colspan=2)

            if mode == 'Histograms':
                for i, data_type in enumerate(HISTOGRAM_DATA_TYPES):
                    hist_data = data[data_type]
                    plot_item = plot_window.addPlot(row=(i // 2) + 1, col=i % 2)
                    plot_item.setTitle(data_type.value.replace('_', ' ').title())
                    plot_item.setLabel('left', 'Count')
                    plot_item.setLabel(
                        'bottom',
                        f"{data_type.value.replace('_', ' ').title()}"
                        f" (Centered at Median {hist_data['median']:.3f})",
                    )
                    plot_item.plot(
                        hist_data['bin_edges'],
                        hist_data['hist'],
                        stepMode=True,
                        fillLevel=0,
                        brush=(100, 100, 255, 150),
                    )
            else:
                plot_dark_cal(plot_window, data)

            plot_widgets.append(plot_window)
        except Exception as e:
            logger.error(f'Error plotting results for {directory}: {e}')

    return plot_widgets
