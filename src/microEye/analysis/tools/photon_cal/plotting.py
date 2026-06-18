import logging

import numpy as np
import pyqtgraph as pg

from microEye.analysis.tools.photon_cal.constants import PLOTS_META, PlotType
from microEye.analysis.tools.photon_cal.models import CalibrationDatasetMeta
from microEye.qt import Qt
from microEye.utils.pyqt2mplt import (
    FigurePayload,
    MatplotlibPlotterDialog,
    PlotSeriesPayload,
    SubplotPayload,
)

logger = logging.getLogger(__name__)


def plot_results(
    datasets: dict[str, CalibrationDatasetMeta],
    returned_data: dict,
    exposure_times_s: np.ndarray,
    use_matplotlib: bool = False,
    use_one_mpl_dialog: bool = True,
) -> list[pg.GraphicsLayoutWidget]:
    plot_widgets: dict[PlotType, pg.PlotWidget] = {}
    # Collect series payloads for optional Matplotlib dialogs (one dialog per plot)
    series_map: dict[PlotType, list[PlotSeriesPayload]] = {pt: [] for pt in PlotType}
    names = [name for name in datasets]
    idx = np.arange(len(names))

    for plot_type in PlotType:
        meta = PLOTS_META[plot_type]
        plot = pg.PlotWidget()
        plot.setWindowTitle(meta['title'])
        plot.plotItem.setTitle(meta['title'])
        plot.plotItem.setLabel('bottom', meta['xlabel'])
        plot.plotItem.setLabel('left', meta['ylabel'])
        if meta.get('legend'):
            plot.plotItem.addLegend()
        if meta.get('grid'):
            plot.plotItem.showGrid(x=True, y=True)
        plot_widgets[plot_type] = plot

        if plot_type == PlotType.QE_BAR:
            qe_bars = pg.BarGraphItem(
                x=idx,
                height=[
                    params['responsivity'] * params['gain_e_per_dn'] * 100
                    for params in returned_data.values()
                ],
                width=0.6,
                name='QE [%]',
                brushes=[pg.intColor(i, hues=len(names)) for i in range(len(names))],
            )
            plot.plotItem.addItem(qe_bars)
            plot.plotItem.getAxis('bottom').setTicks([list(zip(idx, names))])
            # Prepare a Matplotlib series for the bar chart
            series_map[plot_type].append(
                PlotSeriesPayload(
                    x=idx,
                    y=[
                        params['responsivity'] * params['gain_e_per_dn'] * 100
                        for params in returned_data.values()
                    ],
                    plot_type='bar',
                    label='QE [%]',
                    dataset='QE',
                    extras={'width': 0.6},
                )
            )

    for i, (name, _dataset) in enumerate(datasets.items()):
        try:
            params = returned_data[name]
            gain_variance = params.get('fit_variance_dn2', params['var_shot_dn2'])
            variance_source = params.get('fit_variance_source', 'shot')

            color = pg.intColor(i, hues=len(names))
            complement_color = pg.mkColor(
                255 - color.red(), 255 - color.green(), 255 - color.blue()
            )

            plot_widgets[PlotType.GAIN].plotItem.plot(
                params['mean_dn'],
                gain_variance,
                symbol='o',
                name=f"{name} (Gain={params['gain_e_per_dn']:.5f} e-/DN)"
                f"(dark noise={params['read_noise_e']:.2f} e-)"
                f"[{variance_source}]",
                pen=None,
                symbolBrush=color,
                symbolPen=None,
            )
            series_map[PlotType.GAIN].append(
                PlotSeriesPayload(
                    x=params['mean_dn'],
                    y=gain_variance,
                    plot_type='scatter',
                    label=name,
                    dataset=name,
                    style={'color': color.name()},
                )
            )
            plot_widgets[PlotType.GAIN].plotItem.plot(
                params['mean_dn'],
                params['mean_dn'] / params['gain_e_per_dn'] + params['read_noise_dn'],
                pen=pg.mkPen(color=complement_color, style=Qt.PenStyle.DashLine),
            )
            series_map[PlotType.GAIN].append(
                PlotSeriesPayload(
                    x=params['mean_dn'],
                    y=(
                        params['mean_dn'] / params['gain_e_per_dn']
                        + params['read_noise_dn']
                    ),
                    plot_type='line',
                    label=f'{name} fit',
                    dataset=f'{name} fit',
                    style={
                        'color': complement_color.name(),
                        'linestyle': 'dashed',
                    },
                )
            )

            plot_widgets[PlotType.RESPONSE].plotItem.plot(
                params['photons_per_pixel'],
                params['mean_dn'][1:],
                symbol='o',
                name=f"{name} (Responsivity={params['responsivity']:.5f} ADU/photon)",
                pen=None,
                symbolBrush=color,
                symbolPen=None,
            )
            series_map[PlotType.RESPONSE].append(
                PlotSeriesPayload(
                    x=params['photons_per_pixel'],
                    y=params['mean_dn'][1:],
                    plot_type='scatter',
                    label=name,
                    dataset=name,
                    style={'color': color.name()},
                )
            )
            plot_widgets[PlotType.RESPONSE].plotItem.plot(
                params['photons_per_pixel'],
                params['responsivity'] * params['photons_per_pixel']
                + params['responsivity_intercept'],
                pen=pg.mkPen(color=complement_color, style=Qt.PenStyle.DashLine),
            )
            series_map[PlotType.RESPONSE].append(
                PlotSeriesPayload(
                    x=params['photons_per_pixel'],
                    y=params['responsivity'] * params['photons_per_pixel']
                    + params['responsivity_intercept'],
                    plot_type='line',
                    label=f'{name} fit',
                    dataset=f'{name} fit',
                    style={
                        'color': complement_color.name(),
                        'linestyle': 'dashed',
                    },
                )
            )

            plot_widgets[PlotType.QE].plotItem.plot(
                params['photons_per_pixel'],
                params['mean_e'][1:],
                symbol='o',
                name=f"{name} (QE={params['qe'] * 100:.2f}%)",
                pen=None,
                symbolBrush=color,
                symbolPen=None,
            )
            series_map[PlotType.QE].append(
                PlotSeriesPayload(
                    x=params['photons_per_pixel'],
                    y=params['mean_e'][1:],
                    plot_type='scatter',
                    label=name,
                    dataset=name,
                    style={'color': color.name()},
                )
            )
            plot_widgets[PlotType.QE].plotItem.plot(
                params['photons_per_pixel'],
                params['qe'] * params['photons_per_pixel'] + params['qe_intercept'],
                pen=pg.mkPen(color=complement_color, style=Qt.PenStyle.DashLine),
            )
            series_map[PlotType.QE].append(
                PlotSeriesPayload(
                    x=params['photons_per_pixel'],
                    y=(
                        params['qe'] * params['photons_per_pixel']
                        + params['qe_intercept']
                    ),
                    plot_type='line',
                    label=f'{name} fit',
                    dataset=f'{name} fit',
                    style={
                        'color': complement_color.name(),
                        'linestyle': 'dashed',
                    },
                )
            )

            var_e = params['var_total_dn2'][1:] * (params['gain_e_per_dn'] ** 2)
            mask = (var_e > 0) & np.isfinite(var_e)

            if not np.any(mask):
                logger.warning(
                    f'Skipping SNR plot for {name}: no valid variance samples.'
                )
                continue

            photons_valid = params['photons_per_pixel'][mask]

            snr = params['mean_e'][1:][mask] / np.sqrt(var_e[mask])
            snr_db = 20 * np.log10(snr)

            snr_db_at_1000 = np.interp(1000, photons_valid, snr_db)

            plot_widgets[PlotType.SNR].plotItem.plot(
                photons_valid,
                snr_db,
                symbol='o',
                name=f'{name} (SNR at flux 1000 ph/cm²/s={snr_db_at_1000:.2f} dB)',
                pen=None,
                symbolBrush=color,
                symbolPen=None,
            )
            series_map[PlotType.SNR].append(
                PlotSeriesPayload(
                    x=photons_valid,
                    y=snr_db,
                    plot_type='scatter',
                    label=name,
                    dataset=name,
                    style={'color': color.name()},
                )
            )
        except Exception as e:
            logger.error(f'Error plotting results for {name}: {e}')

    # Optionally spawn one Matplotlib dialog per plot and attach it to the
    # corresponding PlotWidget so the dialog isn't garbage-collected.
    if use_matplotlib:
        subplots = []
        for idx, (plot_type, plot) in enumerate(plot_widgets.items()):
            series_list = series_map.get(plot_type, [])
            if not series_list:
                continue
            meta = PLOTS_META[plot_type]
            subplot = SubplotPayload(
                row=0 if not use_one_mpl_dialog else idx // 3,
                col=0 if not use_one_mpl_dialog else idx % 3,
                title=meta.get('title', ''),
                xlabel=meta.get('xlabel', ''),
                ylabel=meta.get('ylabel', ''),
                series=series_list,
                metadata={
                    'dataset_names': names if plot_type == PlotType.QE_BAR else [],
                    'title_alignment': 'left',
                    'title_bold': True,
                    'legend': False,
                },
            )
            if use_one_mpl_dialog:
                subplots.append(subplot)
                continue

            fig = FigurePayload(
                title=meta.get('title', ''),
                subplots=[subplot],
                metadata={'layout': {'rows': 1, 'cols': 1}},
            )
            dlg = MatplotlibPlotterDialog()
            dlg.set_payload(fig)
            dlg.show()
            # Keep a reference on the originating widget to prevent GC.
            plot.mpl_dialog = dlg

        if use_one_mpl_dialog and subplots:
            fig = FigurePayload(
                title='',
                subplots=subplots,
                metadata={
                    'layout': {'rows': 2, 'cols': 3, 'width_in': 15, 'height_in': 10}
                },
            )
            dlg = MatplotlibPlotterDialog()
            dlg.set_payload(fig)
            dlg.show()
            # Attach the dialog to the first plot widget to prevent GC.
            first_plot = next(iter(plot_widgets.values()))
            first_plot.mpl_dialog = dlg

    return list(plot_widgets.values())
