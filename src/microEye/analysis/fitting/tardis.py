import time
import traceback

import numba as nb
import numpy as np
import pyqtgraph as pg

from microEye.analysis.fitting.processing import tardis_data
from microEye.qt import QDateTime, Qt, QtWidgets, Signal
from microEye.utils.thread_worker import thread_worker


class TARDIS_Widget(QtWidgets.QWidget):
    startWorker = Signal(thread_worker)

    def __init__(
            self,
            frames: np.ndarray,
            locX: np.ndarray,
            locY: np.ndarray,
            locZ: np.ndarray=None,
            parent=None,
            ):
        super().__init__(parent=parent)

        self.setWindowTitle('TARDIS Analysis')
        self.frames = frames
        self.locX = locX
        self.locY = locY
        self.locZ = locZ
        self.res = None

        self.hlayout = QtWidgets.QHBoxLayout()
        self.setLayout(self.hlayout)

        self.histogram = pg.PlotWidget()
        self.hlayout.addWidget(self.histogram)

        self.vlayout = QtWidgets.QVBoxLayout()
        self.hlayout.addLayout(self.vlayout)

        self.tardis_group = QtWidgets.QGroupBox('TARDIS parameters')
        self.vlayout.addWidget(self.tardis_group)
        self.tardis_lay = QtWidgets.QFormLayout()
        self.tardis_group.setLayout(self.tardis_lay)

        self.dt = QtWidgets.QComboBox()
        self.dt.setEditable(True)
        self.dt.addItem('[1, 30, 100, 300, 1000]')
        self.dt.addItem('np.linspace(1, 10, 10, dtype=np.int32)')
        self.dt.setToolTip(
            'Enter here a Python expression that provides a list|array of dt values.')

        self.tardis_lay.addRow(
            QtWidgets.QLabel('Δt [frames]:'), self.dt)

        self.exposure = QtWidgets.QDoubleSpinBox()
        self.exposure.setDecimals(4)
        self.exposure.setMinimum(1)
        self.exposure.setMaximum(1e10)
        self.exposure.setValue(50)

        self.tardis_lay.addRow(
            QtWidgets.QLabel('Exposure [ms]:'), self.exposure)

        self.bins = QtWidgets.QSpinBox()
        self.bins.setMinimum(5)
        self.bins.setMaximum(1e4)
        self.bins.setValue(1200)

        self.tardis_lay.addRow(
            QtWidgets.QLabel('N of bins:'), self.bins)


        self.maxDist = QtWidgets.QSpinBox()
        self.maxDist.setMinimum(5)
        self.maxDist.setMaximum(1e4)
        self.maxDist.setValue(1200)

        self.tardis_lay.addRow(
            QtWidgets.QLabel('Max Distance:'), self.maxDist)

        self.clear_plot = QtWidgets.QCheckBox('Clear plot?')
        self.clear_plot.setChecked(True)
        self.tardis_lay.addWidget(self.clear_plot)

        self.compute_btn = QtWidgets.QPushButton('Compute', clicked=self.compute)
        self.tardis_lay.addWidget(self.compute_btn)

        self.fitgroup = QtWidgets.QGroupBox('Fitting parameters')
        self.vlayout.addWidget(self.fitgroup)

        self.fitlay = QtWidgets.QFormLayout()
        self.fitgroup.setLayout(self.fitlay)


        self.res_guess = QtWidgets.QCheckBox('Result as initial guess.')
        self.fitlay.addWidget(self.res_guess)

        self.fit_btn = QtWidgets.QPushButton('Fit', clicked=self.fit_data)
        self.fitlay.addWidget(self.fit_btn)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.log.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.fitlay.addRow(self.log)

        label_style = {'color': '#EEE', 'font-size': '10pt'}

        legend = self.histogram.addLegend()
        legend.anchor(itemPos=(1, 0), parentPos=(1, 0))
        self.histogram.setLabel('bottom', 'Distance [nm]', **label_style)
        self.histogram.setLabel('left', 'PDF', **label_style)
        # self._fit_ref = self.histogram.plot(
        #     _bins, np.zeros_like(_bins), pen=greenP, name='Fit')

    def fit_data(self):
        self.log.appendPlainText(
            QDateTime.currentDateTime().toString(
                '>>> yyyy/MM/dd hh:mm:ss') + ' TARDIS fit started \n'
        )

    def compute(self):
        self.compute_btn.setDisabled(True)
        self.log.appendPlainText(
            QDateTime.currentDateTime().toString(
                '>>> yyyy/MM/dd hh:mm:ss') + ' TARDIS compute started \n'
        )

        start = time.perf_counter_ns()

        def work_func():
            try:
                return tardis_data(
                        self.frames, self.locX, self.locY, self.locZ,
                        dts=self.get_dts(),
                        bins=self.bins.value(), range=self.maxDist.value())
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            self.compute_btn.setDisabled(False)
            if results is not None:
                self.distances, self.edges = results
                stop = time.perf_counter_ns()
                calc = stop - start
                message = f'Computed in {calc/1e9:.5f}s.\n'
                print(message)
                self.log.appendPlainText('    ' + message)
                self.plot_hist()
            else:
                self.log.appendPlainText('    Something went wrong.')


        worker = thread_worker(
            work_func,
            progress=False, z_stage=False)
        worker.signals.result.connect(done)
        self.startWorker.emit(worker)

    def get_dts(self):
        try:
            ldict = {}
            exec(
                'dts = ' + self.dt.currentText(),
                globals(), ldict)
            return ldict['dts']
        except Exception:
            return []

    def plot_hist(self):
        # Clear plot widget
        if self.clear_plot.isChecked():
            self.histogram.clear()

        greenP = pg.mkPen(color='g')
        redP = pg.mkPen(color='r')
        yellowP = pg.mkPen(color='y')
        whiteP = pg.mkPen(color='w')

        dts=self.get_dts()
        # Update histogram
        for idx in range(self.distances.shape[0]):
            self.histogram.plot(
                self.edges, self.distances[idx], stepMode='center',
                fillLevel=0, fillOutline=True,
                pen=pg.mkPen(
                    color=tuple(np.random.choice(range(127, 255),size=3))),
                name=f'Δt = {dts[idx]:d}')
        # TBA: Update fitting curve
        # self._fit_ref = self.histogram.plot(
        #     _bins, np.zeros_like(_bins), pen=greenP, name='Fit')


def get_bincenters(edges: np.ndarray):
    '''Bin centers of histogram bin edges.

    Parameters
    ----------
    edges : np.ndarray
        array of histogram bin edges.

    Returns
    -------
    np.ndarray
        array of bin centers of histogram bin edges.
    '''
    return np.array([(edges[i]+edges[i+1])/2. for i in range(len(edges)-1)])
