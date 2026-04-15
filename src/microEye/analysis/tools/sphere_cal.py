import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from weakref import ref

import cv2
import numpy as np
import ome_types as ome
import pandas as pd
import pyqtgraph as pg
import scipy.constants as sc
import tifffile as tf
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from microEye import __version__
from microEye.images import ImageSequenceBase, TiffSeqHandler, create_array
from microEye.qt import (
    Qt,
    QtCore,
    QtGui,
    QtWidgets,
    getExistingDirectory,
    getOpenFileName,
    getSaveFileName,
)
from microEye.utils.display import DisplayManager, fast_autolevels_opencv
from microEye.utils.thread_worker import QThreadWorker

logger = logging.getLogger(__name__)


def _find_best_linear_segment(
    x: np.ndarray,
    y: np.ndarray,
    min_points: int = 5,
    r2_threshold: float = 0.9999,
):
    '''
    Find longest contiguous segment with high linearity.
    Returns (start_idx, end_idx, model, r2) in sorted x-space.
    '''
    n = len(x)
    if n < min_points:
        raise ValueError('Not enough points for linearity assessment.')

    best = None  # (length, r2, i, j, model)
    i = 0
    for j in range(i + min_points, n + 1):
        xx = x[i:j].reshape(-1, 1)
        yy = y[i:j]
        model = LinearRegression().fit(xx, yy)
        r2 = model.score(xx, yy)
        if r2 >= r2_threshold:
            length = j - i
            cand = (length, r2, i, j, model)
            if (
                best is None
                or (cand[0] > best[0])
                or (cand[0] == best[0] and cand[1] > best[1])
            ):
                best = cand

    if best is None:
        # fallback: best R² among windows of min_points
        best_r2 = -np.inf
        best_pack = None
        i = 0
        for j in range(i + min_points, n + 1):
            xx = x[i:j].reshape(-1, 1)
            yy = y[i:j]
            model = LinearRegression().fit(xx, yy)
            r2 = model.score(xx, yy)
            if r2 > best_r2:
                best_r2 = r2
                best_pack = (min_points, r2, i, j, model)
        best = best_pack

    _, r2, i, j, model = best
    return i, j, model, r2


def _perform_calibration_analysis(data: pd.DataFrame, diameter=10, event=None):
    """
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the columns 'laser_power', 'port', and 'camera'.
    diameter : float
        Diameter of the photodiode in mm. Default is 10 mm.
    """
    logger.info('Running sphere power calibration...')

    results = {
        'power_to_port': None,
        'port_to_camera': None,
    }

    try:
        power = data['laser_power'].to_numpy()  # in mW

        area = np.pi * (diameter / 2) ** 2 / 100  # in cm^2

        irradiance_port = data['port'].to_numpy() / area  # in mW/cm^2

        irradiance_camera = data['camera'].to_numpy() / area  # in uW/cm^2

        # convert camera irradiance to mW/cm^2 for fitting
        irradiance_camera = irradiance_camera / 1000  # in mW/cm^2

        # assess linearity of power vs irradiance_port
        # fit the linear part only of the curve power vs irradiance_port
        x = power
        y = irradiance_port

        # clean + sort
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        i0, i1, model, r2 = _find_best_linear_segment(
            x, y, min_points=5, r2_threshold=0.9999
        )

        results['power_to_port'] = model

        slope = float(model.coef_[0])
        intercept = float(model.intercept_)

        logger.info(
            f'Linear segment: idx[{i0}:{i1}] '
            f'power=[{x[i0]:.4g}, {x[i1 - 1]:.4g}] mW | '
            f'slope={slope:.6g}, intercept={intercept:.6g}, R²={r2:.6f}'
        )

        # fit irradiance_port vs irradiance_camera
        mask = np.isfinite(irradiance_port) & np.isfinite(irradiance_camera)
        irradiance_port = irradiance_port[mask]
        irradiance_camera = irradiance_camera[mask]

        model = LinearRegression().fit(
            irradiance_port.reshape(-1, 1), irradiance_camera
        )

        results['port_to_camera'] = model

        slope_pc = float(model.coef_[0])
        intercept_pc = float(model.intercept_)

        logger.info(
            f'Port to camera fit: slope={slope_pc:.6g}, '
            f'intercept={intercept_pc:.6g}, '
            f'R²={model.score(irradiance_port.reshape(-1, 1), irradiance_camera):.6f}'
        )

        logger.info('Calibration complete.')
        return results
    except Exception as e:
        logger.error(f'Error during calibration analysis: {e}')
        return None


class SphereCalibrationTool(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.sphere_file_path = None
        self.__plot_window = None
        self._cal = None

        self._setup_plots()

        self._setup_ui()

        self._setup_connections()

    def _setup_ui(self):
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        layout = QtWidgets.QFormLayout(self)

        # Sphere power file selection
        self.sphere_file_button = QtWidgets.QPushButton('Select Sphere Power File')
        self.sphere_file_label = QtWidgets.QLabel('No file selected')
        layout.addRow('Sphere Power File:', self.sphere_file_button)
        layout.addRow('', self.sphere_file_label)

        # Detector diameter
        self.diameter_spinbox = QtWidgets.QDoubleSpinBox()
        self.diameter_spinbox.setRange(0.1, 100.0)

        self.diameter_spinbox.setValue(10.0)
        self.diameter_spinbox.setSuffix(' mm')

        layout.addRow('Photo-Diode Diameter:', self.diameter_spinbox)

        # Run analysis button
        self.run_button = QtWidgets.QPushButton('Run Sphere Power Calibration')
        layout.addRow('', self.run_button)

        # convert from laser power to port irradiance
        # accept input field
        self.convert_input_field = QtWidgets.QDoubleSpinBox()
        self.convert_input_field.setRange(0.0, 1000.0)
        self.convert_input_field.setDecimals(6)
        self.convert_input_field.setToolTip(
            'Enter power in mW or port irradiance in mW/cm²'
        )
        layout.addRow('Conversion Input:', self.convert_input_field)

        self.convert_output_label = QtWidgets.QLabel('')
        layout.addRow('Conversion Output:', self.convert_output_label)

        self.power_to_port_button = QtWidgets.QPushButton(
            'Laser Power to Port Irradiance'
        )
        self.power_to_port_button.clicked.connect(self._convert_power_to_port)
        layout.addRow('', self.power_to_port_button)

        # convert from port irradiance to camera irradiance
        self.port_to_camera_button = QtWidgets.QPushButton('Port to Camera Irradiance')
        self.port_to_camera_button.clicked.connect(self._convert_port_to_camera)
        layout.addRow('', self.port_to_camera_button)

        # Results display (will be populated after analysis)
        self.results_groupbox = QtWidgets.QGroupBox('Calibration Results')
        self.results_layout = QtWidgets.QVBoxLayout(self.results_groupbox)
        layout.addRow(self.results_groupbox)

        self.fit_label = QtWidgets.QLabel('')
        self.results_layout.addWidget(self.fit_label)

        self.results_layout.addWidget(self.__plot_window)

    def _setup_plots(self):
        self.__plot_window: pg.GraphicsLayout = pg.GraphicsLayoutWidget(
            title='Sphere Power Calibration Results',
        )

        # Plot power vs port with linear fit
        self.__p1: pg.PlotItem = self.__plot_window.addPlot(
            row=0,
            col=0,
        )
        self.__p1.addLegend()  # Add legend to the first plot
        self.__p1.setTitle('Port Irradiance vs Laser Power')
        self.__p1.setLabel('left', 'Port Irradiance [mW/cm²]')
        self.__p1.setLabel('bottom', 'Laser Power [mW]')

        # Plot port vs camera with linear fit
        self.__p2: pg.PlotItem = self.__plot_window.addPlot(
            row=1,
            col=0,
        )
        self.__p2.addLegend()  # Add legend to the second plot
        self.__p2.setTitle('Camera Irradiance vs Port Irradiance')
        self.__p2.setLabel('left', 'Camera Irradiance [mW/cm²]')
        self.__p2.setLabel('bottom', 'Port Irradiance [mW/cm²]')

    def _setup_connections(self):
        self.sphere_file_button.clicked.connect(self._select_sphere_file)
        self.run_button.clicked.connect(self._run_calibration)

    def _get_data(self, filename=None):
        if filename is None:
            filename = self.sphere_file_path

        if not filename:
            logger.error('No sphere power file selected.')
            return None

        # Load the data
        if filename.endswith('.csv') or filename.endswith('.txt'):
            return pd.read_csv(filename)
        elif filename.endswith('.h5') or filename.endswith('.hdf'):
            return pd.read_hdf(filename)
        elif filename.endswith('.tsv'):
            return pd.read_csv(filename, delimiter='\t')
        else:
            logger.error('Unsupported file format.')
            return None

    def _select_sphere_file(self):
        file_path, _ = getOpenFileName(
            self,
            caption='Select Sphere Power File',
            filter='CSV Files (*.csv);;HDF Files (*.h5);;All Files (*)',
        )
        if file_path:
            needed_columns = ['laser_power', 'port', 'camera']

            data = self._get_data(filename=file_path)

            if not all(col in data.columns for col in needed_columns):
                logger.error(
                    'Selected file is missing required columns.'
                    f'Needed: {needed_columns}'
                )
                return

            self.sphere_file_label.setText(os.path.basename(file_path))
            self.sphere_file_path = file_path

    def _plot_calibration_results(self, data: pd.DataFrame, results, diameter=10):
        # Clear previous plots
        self.__p1.clear()
        self.__p2.clear()

        self.__p1.plot(
            data['laser_power'],
            data['port'] / (np.pi * (diameter / 2) ** 2 / 100),
            pen=None,
            symbol='o',
            name='Data Points',
        )
        if results['power_to_port'] is not None:
            model = results['power_to_port']
            x_fit = np.linspace(
                data['laser_power'].min(), data['laser_power'].max(), 100
            )
            y_fit = model.predict(x_fit.reshape(-1, 1))
            self.__p1.plot(x_fit, y_fit, pen='r', name=f'Linear fit')

        # Plot port vs camera with linear fit
        self.__p2.plot(
            data['port'] / (np.pi * (diameter / 2) ** 2 / 100),
            data['camera'] / (np.pi * (diameter / 2) ** 2 / 100) / 1000,
            pen=None,
            symbol='o',
            name='Data Points',
        )
        if results['port_to_camera'] is not None:
            model = results['port_to_camera']
            x_fit = np.linspace(
                data['port'].min() / (np.pi * (diameter / 2) ** 2 / 100),
                data['port'].max() / (np.pi * (diameter / 2) ** 2 / 100),
                100,
            )
            y_fit = model.predict(x_fit.reshape(-1, 1))
            self.__p2.plot(x_fit, y_fit, pen='r', name=f'Linear fit')

            x_points = float(results['power_to_port'].coef_[0]) * np.array(
                [13.5, 11.8, 14.5, 5.5, 15, 6]
            )
            y_points = model.predict(x_points.reshape(-1, 1))
            self.__p2.plot(
                x_points,
                y_points,
                pen=None,
                # fill yellow to distinguish from original data points
                brush=pg.mkBrush(255, 255, 0, 100),
                symbol='+',
                name='Predicted Values',
            )

    def _run_calibration(self):
        if not hasattr(self, 'sphere_file_path'):
            logger.error('No sphere power file selected.')
            return

        # Load the data
        data = self._get_data()

        diameter = self.diameter_spinbox.value()

        self.run_button.setEnabled(False)

        def analysis_finished(result):
            self.run_button.setEnabled(True)
            logger.info('Calibration analysis finished.')

            if result is not None:
                self._plot_calibration_results(data, result, diameter=diameter)
                self._cal = result

                # Update results label
                if result['power_to_port'] is not None:
                    slope = float(result['power_to_port'].coef_[0])
                    intercept = float(result['power_to_port'].intercept_)
                    self.fit_label.setText(
                        f'Power to Port Fit: slope={slope:.6g}, '
                        f'intercept={intercept:.6g}'
                    )

                if result['port_to_camera'] is not None:
                    slope_pc = float(result['port_to_camera'].coef_[0])
                    intercept_pc = float(result['port_to_camera'].intercept_)
                    self.fit_label.setText(
                        self.fit_label.text()
                        + f'\nPort to Camera Fit: slope={slope_pc:.6g}, '
                        f'intercept={intercept_pc:.6g}'
                    )

        # Run analysis in a separate thread to avoid blocking the UI
        worker = QThreadWorker(_perform_calibration_analysis, data, diameter)
        worker.signals.result.connect(analysis_finished)

        QtCore.QThreadPool.globalInstance().start(worker)

    def _convert_power_to_port(self):
        try:
            power = self.convert_input_field.value()
            port_irradiance = self.power_to_port_irradiance([[power]])[0]
        except Exception as e:
            logger.error(f'Error converting power to port irradiance: {e}')
            self.convert_output_label.setText('Conversion error')
            return

        self.convert_output_label.setText(f'{port_irradiance:.6g} mW/cm²')
        logger.info(
            f'Power {power} mW corresponds to '
            f'Port Irradiance {port_irradiance:.6g} mW/cm²'
        )

    def _convert_port_to_camera(self):
        try:
            port_irradiance = self.convert_input_field.value()
            camera_irradiance = self.port_to_camera_irradiance([[port_irradiance]])[0]
        except Exception as e:
            logger.error(f'Error converting port to camera irradiance: {e}')
            self.convert_output_label.setText('Conversion error')
            return

        logger.info(
            f'Port Irradiance {port_irradiance:.6g} mW/cm² corresponds to '
            f'Camera Irradiance {camera_irradiance * 1000:.6g} uW/cm²'
        )
        self.convert_output_label.setText(f'{camera_irradiance * 1000:.6g} uW/cm²')

    def port_to_camera_irradiance(self, port_irradiance):
        if self._cal is None or self._cal['port_to_camera'] is None:
            raise ValueError('Calibration not performed yet.')

        model = self._cal['port_to_camera']
        return model.predict(np.array(port_irradiance).reshape(-1, 1)).flatten()

    def port_power_to_camera_irradiance(self, port_power):
        area = np.pi * (self.diameter_spinbox.value() / 2) ** 2 / 100  # in cm^2

        irradiance_port = port_power / area  # in mW/cm^2

        return self.port_to_camera_irradiance(irradiance_port)

    def power_to_port_irradiance(self, power):
        if self._cal is None or self._cal['power_to_port'] is None:
            raise ValueError('Calibration not performed yet.')

        model = self._cal['power_to_port']
        return model.predict(np.array(power).reshape(-1, 1)).flatten()

    def power_to_camera_irradiance(self, power):
        port_irradiance = self.power_to_port_irradiance(power)
        return self.port_to_camera_irradiance(port_irradiance)
