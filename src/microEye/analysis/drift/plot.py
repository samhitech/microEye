import logging

import numpy as np
import pyqtgraph as pg

logger = logging.getLogger(__name__)


def plot_drift(
    frames_new,
    x,
    y,
    z=None,
    error_x=None,
    error_y=None,
    error_z=None,
    title='Drift Estimation',
    subtitle='',
):
    logger.debug(f'Plotting {title} ...')

    # plot results
    plt = pg.plot()

    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties of the label for y axis
    plt.setLabel('left', 'drift', units='nm')

    # set properties of the label for x axis
    plt.setLabel('bottom', 'frame', units='')

    plt.setWindowTitle(title)

    # plot subtitle
    if subtitle:
        plt.getPlotItem().setTitle(subtitle)

    # setting horizontal range
    plt.setXRange(0, np.max(frames_new))

    # setting vertical range
    # plt.setYRange(0, 1)

    # Plot x-drift with error bars
    errors = []
    error_pens = []
    if error_x is not None:
        errors.append(error_x)
        error_pens.append(pg.mkPen(color=(255, 0, 0, 24)))
    if error_y is not None:
        errors.append(error_y)
        error_pens.append(pg.mkPen(color=(255, 255, 0, 24)))
    if error_z is not None:
        errors.append(error_z)
        error_pens.append(pg.mkPen(color=(0, 0, 255, 24)))

    # Plot x-drift with error bars
    for error, pen in zip(errors, error_pens):
        if error is not None:
            err = pg.ErrorBarItem(
                x=frames_new,
                y=x if error is error_x else y if error is error_y else z,
                top=error,
                bottom=error,
                beam=0.5,
                pen=pen,
            )
            plt.addItem(err)

    # Plot drift curves
    plot = []
    plot_pens = []
    plot_names = []
    if x is not None:
        plot.append(x)
        plot_pens.append(pg.mkPen(color='r', width=2))
        plot_names.append('x-drift')
    if y is not None:
        plot.append(y)
        plot_pens.append(pg.mkPen(color='y', width=2))
        plot_names.append('y-drift')
    if z is not None:
        plot.append(z)
        plot_pens.append(pg.mkPen(color='g', width=2))
        plot_names.append('z-drift')

    for curve, pen, name in zip(plot, plot_pens, plot_names):
        plt.plot(frames_new, curve, pen=pen, symbol=None, symbolBrush=0.2, name=name)

    logger.debug('Drift plotting done.')
