import sys
import traceback
import typing

import numpy as np
import pandas as pd
import pyqtgraph as pg

from microEye.qt import QApplication, QtWidgets, Signal


class resultsStatsWidget(QtWidgets.QWidget):
    dataFilterUpdated = Signal(pd.DataFrame)

    def __init__(self, parent: typing.Optional['QtWidgets.QWidget'] = None):
        super().__init__(parent=parent)

        self._minHeight = 125

        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self.plot_widgets = []
        self.plot_lr = []
        self.filtered: pd.DataFrame = None

        self._extended = False

    def toggle_track_plots(self):
        self._extended = not self._extended
        self.setData(self._df)

    def setData(self, df: pd.DataFrame):
        self.clear()

        if df is None:
            return

        self._df = df

        for _idx, column in enumerate(df.columns):
            pw, lr = self._create_plot_widget(column, df)
            self.plot_widgets.append(pw)
            self.plot_lr.append(lr)
            self._layout.addWidget(pw)
            lr.sigRegionChangeFinished.connect(self.update)

        self._derive_tracking_df()

        if self._tracking_df is not None:
            for column in self._tracking_df.columns:
                pw, lr = self._create_plot_widget(column, self._tracking_df)
                self.plot_widgets.append(pw)
                self.plot_lr.append(lr)
                self._layout.addWidget(pw)
                lr.sigRegionChangeFinished.connect(self.update)

    def _derive_tracking_df(self):
        if self._df['trackID'].max() > 0 and self._extended:
            track_ids = self._df[self._df['trackID'] > 0]['trackID'].to_numpy()
            distances = self._df[self._df['trackID'] > 0][
                'neighbour_distance'
            ].to_numpy()

            # Get unique track IDs and their counts
            unique_ids, counts = np.unique(track_ids, return_counts=True)

            # Calculate sum of distances per track using bincount
            sums = np.bincount(track_ids.astype(int), weights=distances)[
                unique_ids.astype(int)
            ]

            # Calculate MSD using bincount for sum of squares
            msd = (
                np.bincount(track_ids.astype(int), weights=distances**2)[
                    unique_ids.astype(int)
                ]
                / counts
            )

            self._tracking_df = pd.DataFrame(
                {'track_length': counts, 'track_distance': sums, 'track_msd': msd},
                index=unique_ids,
            )
        else:
            self._tracking_df = None

    def _create_plot_widget(
        self, column: str, df: pd.DataFrame
    ) -> tuple[pg.PlotWidget, pg.LinearRegionItem]:
        '''
        Create a plot widget with appropriate visualization for a dataframe column.
        '''
        # Create basic plot widget
        pw = pg.PlotWidget(self)
        pw.setMinimumHeight(self._minHeight)
        pw.setLabel('left', 'Counts', units='')
        pw.setLabel('bottom', column, units='')

        # Create linear region item
        bounds = [df[column].min(), df[column].max()]
        lr = pg.LinearRegionItem(bounds, bounds=bounds, movable=True)
        pw.addItem(lr)

        # Add visualization based on column type
        if column == 'trackID':
            self._add_trackID_visualization(pw, df[column])
        else:
            self._add_histogram_visualization(pw, lr, df[column], column)

        return pw, lr

    def _add_trackID_visualization(
        self, plot_widget: pg.PlotWidget, data_series: pd.Series
    ):
        '''Add bar graph visualization for trackID column.'''
        data = data_series.to_numpy()
        data = data[data.nonzero()]

        if len(data) > 0:
            uniq, counts = np.unique(data, return_counts=True)
            bars = pg.BarGraphItem(x=uniq, height=counts, width=0.9, brush='b')
            plot_widget.addItem(bars)
        else:
            plot_widget.plot(
                [0], [1], fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150)
            )

    def _add_histogram_visualization(
        self,
        plot_widget: pg.PlotWidget,
        lr: pg.LinearRegionItem,
        data_series: pd.Series,
        column_name: str,
    ):
        '''Add histogram visualization for numeric columns.'''
        try:
            min_val, max_val = self._calculate_column_range(data_series, column_name)

            counts = data_series.count()
            hist, bins = np.histogram(
                data_series.to_numpy(),
                bins=min(counts + 1, 1024),
                range=(min_val, max_val),
            )

            if column_name != 'neighbour_distance':
                lr.setRegion((min_val, max_val))

            plot_widget.plot(
                bins,
                hist / np.max(hist),
                stepMode='center',
                fillLevel=0,
                fillOutline=True,
                brush=(0, 0, 255, 150),
            )
        except Exception:
            traceback.print_exc()
            print(column_name)

    def _calculate_column_range(
        self, data_series: pd.Series, column_name: str
    ) -> tuple[float, float]:
        '''Calculate appropriate min and max values for column visualization.'''
        if column_name in [
            'frame',
            'x',
            'y',
            'z',
            'iteration',
            'track_length',
            'track_distance',
            'track_msd',
        ]:
            min_val = data_series.min()
            max_val = data_series.max()
        elif column_name == 'neighbour_distance':
            filtered_data = data_series[data_series > 0]
            min_val = filtered_data.min() if not pd.isna(filtered_data.min()) else 0
            max_val = filtered_data.max() if not pd.isna(filtered_data.max()) else 1
        else:
            q1, q3 = np.nanpercentile(data_series.to_numpy(), [10, 90])
            iqr = q3 - q1
            min_val = (
                q1 - 1.5 * iqr if column_name == 'loglike' else max(q1 - 1.6 * iqr, 0)
            )
            max_val = q3 + 1.6 * iqr

        return min_val, max_val

    def _clearLayout(self):
        for i in reversed(range(self._layout.count())):
            widgetToRemove = self._layout.itemAt(i).widget()
            # remove it from the layout list
            self._layout.removeWidget(widgetToRemove)
            # remove it from the gui
            widgetToRemove.setParent(None)

    def clear(self):
        self.plot_widgets = []
        self.plot_lr: list[pg.LinearRegionItem] = []

        self._df = None
        self._tracking_df = None

        self._clearLayout()

    def update(self):
        mask = np.ones(self._df.count().max(), dtype=bool)
        for idx, column in enumerate(self._df.columns):
            Rmin, Rmax = self.plot_lr[idx].getRegion()

            if Rmin >= Rmax:
                continue

            column_data = self._df[column].to_numpy()
            mask &= (column_data >= Rmin) & (column_data <= Rmax)

        if self._tracking_df is not None and self._extended:
            tr_mask = np.ones(self._tracking_df.count().max(), dtype=bool)
            for idx, column in enumerate(self._tracking_df.columns):
                Rmin, Rmax = self.plot_lr[idx + len(self._df.columns)].getRegion()

                if Rmin >= Rmax:
                    continue

                column_data = self._tracking_df[column].to_numpy()
                tr_mask &= (column_data >= Rmin) & (column_data <= Rmax)

            # Create lookup array for track filtering
            track_lookup = np.zeros(int(self._df['trackID'].max()) + 1, dtype=bool)
            track_lookup[self._tracking_df.index[tr_mask].astype(int)] = True

            # Apply track filter to main mask
            track_ids = self._df['trackID'].to_numpy()
            valid_tracks = (track_ids > 0) & track_lookup[track_ids.astype(int)]
            mask &= valid_tracks

        self.filtered = self._df[mask]

        self.dataFilterUpdated.emit(self.filtered)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = resultsStatsWidget()
    win.show()
    win.setData(5)

    app.exec()
