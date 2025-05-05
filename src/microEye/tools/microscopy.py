import sys
from enum import Enum, auto
from typing import Optional, Union

import numpy as np
import pyqtgraph as pg
from tabulate import tabulate

from microEye.qt import QAction, QApplication, QtCore, QtWidgets


class Immersion(Enum):
    AIR = 1.000
    WATER = 1.333
    OIL = 1.515
    GLYCERIN = 1.473
    SILICONE = 1.410

    @classmethod
    def list(cls):
        return [immersion.name for immersion in cls]

    @classmethod
    def get(cls, value: Union[str, float]):
        if isinstance(value, float):
            for immersion in cls:
                if immersion.value == value:
                    return immersion
            raise ValueError(
                f"Immersion medium with refractive index '{value}' not found."
            )
        elif isinstance(value, str):
            name = value.upper()
            if name in cls.__members__:
                return cls[name]
            else:
                raise ValueError(f"Immersion medium '{name}' not found.")
        else:
            raise TypeError('Value must be a string or a float.')


class Manufacturer(Enum):
    NIKON = auto()
    OLYMPUS = auto()
    LEICA = auto()
    ZEISS = auto()
    MITUTOYO = auto()
    OTHER = auto()

    @classmethod
    def list(cls):
        return [manufacturer.name for manufacturer in cls]

    @classmethod
    def get(cls, name):
        name = name.upper()
        if name in cls.__members__:
            return cls[name]
        else:
            raise cls.OTHER


TUBE_LENS = {
    Manufacturer.NIKON: 200,
    Manufacturer.OLYMPUS: 180,
    Manufacturer.LEICA: 200,
    Manufacturer.ZEISS: 165,
    Manufacturer.MITUTOYO: 200,
    Manufacturer.OTHER: -1,
}

PARFOCAL_DISTANCE = {
    Manufacturer.NIKON: 60,
    Manufacturer.OLYMPUS: 45,
    Manufacturer.LEICA: 45,
    Manufacturer.ZEISS: 45,
    Manufacturer.MITUTOYO: 95,
    Manufacturer.OTHER: -1,
}

PIXEL_SIZE = [2.74, 3.45, 4.54, 5.86, 6.5, 9, 11, 13, 13.5, 16]


class Objective:
    def __init__(
        self,
        manufacturer: Union[str, Manufacturer],
        magnification: float,
        numerical_aperture: float,
        immersion_medium: Union[str, Immersion, float] = Immersion.AIR,
    ):
        self.manufacturer = (
            manufacturer
            if isinstance(manufacturer, Manufacturer)
            else Manufacturer.get(manufacturer)
        )
        if magnification <= 0:
            raise ValueError('Magnification must be a positive number.')
        self.magnification = magnification

        self.immersion_medium = (
            immersion_medium
            if isinstance(immersion_medium, Immersion)
            else Immersion.get(immersion_medium)
            if isinstance(immersion_medium, (str, float))
            else Immersion.AIR
        )

        if numerical_aperture <= 0 or numerical_aperture > self.immersion_medium.value:
            raise ValueError(
                'Numerical aperture must be a positive number and less than or equal '
                + 'to the immersion medium refractive index.'
            )

        self.numerical_aperture = numerical_aperture

    @property
    def theta(self) -> float:
        '''Half-angle of the cone of light in radians.'''
        return np.arcsin(self.numerical_aperture / self.immersion_medium.value)

    @property
    def critical_angle(self) -> float:
        '''Critical angle for total internal reflection in radians.'''
        if (
            self.immersion_medium != Immersion.AIR
            and self.immersion_medium.value > Immersion.WATER.value
        ):
            return np.arcsin(Immersion.WATER.value / self.immersion_medium.value)

        return np.nan

    @property
    def n(self) -> float:
        '''Refractive index of the immersion medium.'''
        return self.immersion_medium.value

    @property
    def tube_lens(self) -> Optional[float]:
        '''Focal length of the tube lens in mm.'''
        return TUBE_LENS.get(self.manufacturer, None)

    @property
    def parfocal_distance(self) -> Optional[float]:
        '''Parfocal distance in mm.'''
        return PARFOCAL_DISTANCE.get(self.manufacturer, None)

    @property
    def focal_length(self) -> float:
        '''Focal length of the objective in mm.'''
        return self.tube_lens / self.magnification if self.tube_lens > 0 else -1.0

    @property
    def pupil_diameter(self) -> float:
        '''Diameter of the pupil in mm.'''
        return 2 * self.focal_length * self.numerical_aperture

    @property
    def collection_angle(self) -> float:
        '''Collection angle in radians.'''
        return 100 * (1 - np.cos(self.theta))

    def rayleigh_criterion(self, wavelength: float) -> float:
        '''Rayleigh criterion in nm.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.

        Returns
        -------
        tuple[float, float]
            Lateral (XY) Rayleigh criterion in nm; Axial (Z) Rayleigh criterion in nm.
        '''
        return (
            0.61 * wavelength / self.numerical_aperture,
            2 * self.n * wavelength / self.numerical_aperture**2,
        )

    def beam_position(self, angle: float) -> float:
        '''Calculate the beam position at the back focal plane.

        Parameters
        ----------
        angle : float
            Angle in radians.

        Returns
        -------
        float
            Beam position in mm.
        '''
        if self.focal_length <= 0:
            raise ValueError(
                'Focal length must be positive to calculate beam position.'
            )
        return self.focal_length * self.n * np.sin(angle)

    def nyquist_limit(self, wavelength: float) -> float:
        '''Nyquist limit in nm.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.

        Returns
        -------
        float
            Nyquist limit in nm.
        '''
        return list(criterion / 2 for criterion in self.rayleigh_criterion(wavelength))

    def field_of_view(
        self, field_number: float, camera_adapter_magnification: float = 1
    ) -> float:
        '''Field of view in mm

        Parameters
        ----------
        field_number : float
            Field number in mm.

        Returns
        -------
        float
            Field of view in um.
        '''
        return 1000 * field_number / (camera_adapter_magnification * self.magnification)

    def optimal_pixel_size(
        self, wavelength: float, camera_adapter_magnification: float = 1
    ) -> float:
        '''Optimal pixel size in nm.

        Parameters
        ----------
        fov : float
            Field of view in mm.
        wavelength : float
            Wavelength in nm.
        camera_adapter_magnification : float
            Camera adapter magnification.

        Returns
        -------
        float
            Optimal pixel size in um.
        '''
        return (
            self.rayleigh_criterion(wavelength)[0]
            / 2000
            * (self.magnification * camera_adapter_magnification)
        )

    def detector_size(self, fov: float, wavelength: float) -> float:
        '''Detector size in pixels.

        Parameters
        ----------
        fov : float
            Field of view in mm.
        wavelength : float
            Wavelength in nm.

        Returns
        -------
        tuple[float, float]
            Lateral (XY) detector size in pixels
        '''
        return fov / self.rayleigh_criterion(wavelength)[0]

    def print(
        self, wavelength: Optional[float] = None, field_number: Optional[float] = None
    ) -> None:
        '''Print the objective properties.'''
        data = [
            ['Manufacturer', self.manufacturer.name],
            ['Magnification', f'{self.magnification:.3f}'],
            ['Numerical aperture', f'{self.numerical_aperture:.3f}'],
            ['Immersion medium', self.immersion_medium.name],
            ['Focal length (mm)', f'{self.focal_length:.3f}'],
            ['Pupil diameter (mm)', f'{self.pupil_diameter:.3f}'],
            ['Collection angle (%)', f'{self.collection_angle:.2f}%'],
        ]

        if wavelength is not None:
            rayleigh_xy, rayleigh_z = self.rayleigh_criterion(wavelength)
            nyquist_xy, nyquist_z = self.nyquist_limit(wavelength)

            data.extend(
                [
                    [
                        f'Rayleigh criterion @ {wavelength}nm (XY, Z) (nm)',
                        f'({rayleigh_xy:.3f}, {rayleigh_z:.3f})',
                    ],
                    [
                        f'Nyquist limit @ {wavelength}nm (XY, Z) (nm)',
                        f'({nyquist_xy:.3f}, {nyquist_z:.3f})',
                    ],
                    [
                        f'Optimal pixel size @ {wavelength}nm (um)',
                        f'{self.optimal_pixel_size(wavelength):.3f}',
                    ],
                ]
            )

            if field_number is not None:
                detector_size = self.detector_size(field_number, wavelength)
                data.append(
                    [
                        f'Detector size @ {wavelength}nm (pixels)',
                        (f'{detector_size:.3f}',) * 2,
                    ]
                )

        print(tabulate(data, headers=['Property', 'Value'], tablefmt='rounded_grid'))


class TIRF:
    @staticmethod
    def critical_angle(
        objective: Objective,
        n1: float = Immersion.OIL.value,
        n2: float = Immersion.WATER.value,
    ) -> float:
        '''Calculate the critical angle for total internal reflection.

        Parameters
        ----------
        objective : Objective
            The objective lens class.
        panetration_depth : float, optional
            The penetration depth in nm, by default 150.
        n1 : float, optional
            The refractive index of the immersion medium, by default Immersion.OIL.
        n2 : float, optional
            The refractive index of the sample, by default Immersion.WATER.
        '''
        # Laser wavelengths
        wavelengths = [405, 488, 532, 561, 638]  # nm

        # Calculate min and max angles
        min_angle = np.round(np.degrees(np.arcsin(n2 / n1)), 1)
        max_angle = np.round(np.degrees(objective.theta), 1)

        max_range = 80  # arbitrary upper limit for demonstration
        angles = np.arange(min_angle, max_range, 0.1)

        # Calculate penetration depths for each wavelength
        wavelength_depths = {}
        for wavelength in wavelengths:
            penetration_depths = []
            for angle in angles:
                sin_theta = np.sin(np.radians(angle))
                term = n1**2 * sin_theta**2 - n2**2
                if term <= 0:
                    penetration_depths.append(
                        np.nan  # Avoid division by zero or sqrt of negative
                    )
                    continue
                value = wavelength / (4 * np.pi * np.sqrt(term))
                penetration_depths.append(value)
            wavelength_depths[wavelength] = np.array(penetration_depths)

        return angles, wavelength_depths


class ObjectiveCalculator(QtWidgets.QDialog):
    COLORS = {
        405: '#8200c8',
        488: '#00f7ff',
        532: '#65ff00',
        561: '#c6ff00',
        638: '#ff2b00',
    }

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle('Objective Parameters')
        self.setGeometry(100, 100, 1000, 700)

        # Main layout
        main_layout = QtWidgets.QGridLayout()

        # Input fields
        input_group = QtWidgets.QGroupBox('Objective Parameters')
        input_layout = QtWidgets.QFormLayout()
        input_group.setLayout(input_layout)
        input_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        input_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        input_layout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(10)
        self.manufacturer_combo = QtWidgets.QComboBox()
        self.manufacturer_combo.addItems(Manufacturer.list())

        self.magnification_input = QtWidgets.QDoubleSpinBox()
        self.magnification_input.setRange(0.1, 1000.0)
        self.magnification_input.setValue(60.0)
        self.magnification_input.setDecimals(2)
        self.magnification_input.setMinimumWidth(100)

        self.numerical_aperture_input = QtWidgets.QDoubleSpinBox()
        self.numerical_aperture_input.setRange(0.1, 2.0)
        self.numerical_aperture_input.setValue(1.49)
        self.numerical_aperture_input.setDecimals(2)
        self.numerical_aperture_input.setMinimumWidth(100)

        self.immersion_combo = QtWidgets.QComboBox()
        self.immersion_combo.addItems(Immersion.list())
        self.immersion_combo.setCurrentText(Immersion.OIL.name)

        self.wavelength_input = QtWidgets.QSpinBox()
        self.wavelength_input.setRange(200, 2000)  # Wavelength in nm
        self.wavelength_input.setValue(550)
        self.wavelength_input.setMinimumWidth(100)

        self.field_number_input = QtWidgets.QDoubleSpinBox()
        self.field_number_input.setRange(0.1, 1000.0)  # Field number in mm
        self.field_number_input.setValue(3.5)
        self.field_number_input.setDecimals(1)
        self.field_number_input.setMinimumWidth(100)

        self.camera_adapter_input = QtWidgets.QDoubleSpinBox()
        self.camera_adapter_input.setRange(0.1, 1000.0)  # Camera adapter magnification
        self.camera_adapter_input.setValue(0.4)
        self.camera_adapter_input.setDecimals(2)
        self.camera_adapter_input.setMinimumWidth(100)

        self.camera_pixel_size_input = QtWidgets.QDoubleSpinBox()
        self.camera_pixel_size_input.setRange(1.0, 100.0)  # Camera pixel size in um
        self.camera_pixel_size_input.setValue(2.74)
        self.camera_pixel_size_input.setDecimals(3)
        self.camera_pixel_size_input.setMinimumWidth(100)

        input_layout.addRow('Manufacturer:', self.manufacturer_combo)
        input_layout.addRow('Magnification:', self.magnification_input)
        input_layout.addRow('Numerical Aperture:', self.numerical_aperture_input)
        input_layout.addRow('Immersion Medium:', self.immersion_combo)
        input_layout.addRow('Wavelength (nm):', self.wavelength_input)
        input_layout.addRow('Field Number (mm):', self.field_number_input)
        input_layout.addRow('Camera Adapter Magnification:', self.camera_adapter_input)
        input_layout.addRow('Camera Pixel Size (um):', self.camera_pixel_size_input)

        main_layout.addWidget(input_group, 0, 0, 1, 1)

        # Update button
        self.update_button = QtWidgets.QPushButton('Update')
        self.update_button.clicked.connect(self.update_parameters)
        main_layout.addWidget(self.update_button, 1, 0, 1, 3)

        # Tabs for derived parameters and graphical illustration
        self.tabs = QtWidgets.QTabWidget()

        # Tab for derived parameters table
        self.table_tab = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout()
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setMinimumWidth(400)
        self.table.setHorizontalHeaderLabels(['Property', 'Value'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        table_layout.addWidget(self.table)
        self.table_tab.setLayout(table_layout)
        self.tabs.addTab(self.table_tab, 'Parameters')

        # Tab for graphical illustration
        self.graph_tab = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        # self.plot_widget.setBackground('w')
        self.plot_widget.setTitle('Rayleigh Criterion', size='12pt')
        self.plot_widget.setLabel('left', 'Criterion (nm)')
        self.plot_widget.setLabel('bottom', 'Wavelength (nm)')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        graph_layout.addWidget(self.plot_widget)
        self.graph_tab.setLayout(graph_layout)
        self.tabs.addTab(self.graph_tab, 'Graph')

        # Tab for TIRF
        self.critical_angle_tab = QtWidgets.QWidget()
        tirf_layout = QtWidgets.QVBoxLayout()
        self.critical_angle_plot: pg.PlotItem = pg.PlotWidget()
        self.critical_angle_plot.setTitle('TIRF Critical Angle', size='12pt')
        self.critical_angle_plot.setLabel('left', 'Panetration Depth (nm)')
        self.critical_angle_plot.setLabel('bottom', 'Angle (degrees)')
        self.critical_angle_plot.showGrid(x=True, y=True)
        self.critical_angle_plot.addLegend(
            offset=(-10, 10),
        )

        tirf_layout.addWidget(self.critical_angle_plot)
        self.critical_angle_tab.setLayout(tirf_layout)
        self.tabs.addTab(self.critical_angle_tab, 'TIRF')

        # Tab for beam angle to beam position @ BFP
        self.beam_angle_tab = QtWidgets.QWidget()
        beam_angle_layout = QtWidgets.QVBoxLayout()
        self.beam_angle: pg.PlotItem = pg.PlotWidget()
        self.beam_angle.setTitle('Beam Angle to Position at BFP', size='12pt')
        self.beam_angle.setLabel('bottom', 'Beam Position @ BFP (mm)')
        self.beam_angle.setLabel('left', 'Angle (degrees)')
        self.beam_angle.showGrid(x=True, y=True)
        self.beam_angle.addLegend()

        beam_angle_layout.addWidget(self.beam_angle)
        self.beam_angle_tab.setLayout(beam_angle_layout)
        self.tabs.addTab(self.beam_angle_tab, 'Beam Angle')

        # Add tabs to the main layout
        main_layout.addWidget(self.tabs, 0, 1, 1, 2)

        # Set main widget
        self.setLayout(main_layout)

        # Initialize Objective
        self.objective = None

    def update_parameters(self):
        # Get user inputs
        manufacturer = Manufacturer.get(self.manufacturer_combo.currentText())
        magnification = self.magnification_input.value()
        numerical_aperture = self.numerical_aperture_input.value()
        immersion_medium = Immersion.get(self.immersion_combo.currentText())
        wavelength = self.wavelength_input.value()
        field_number = self.field_number_input.value()

        # Create Objective instance
        try:
            self.objective = Objective(
                manufacturer, magnification, numerical_aperture, immersion_medium
            )

            # Update table with derived parameters
            self.update_table(wavelength, field_number)

            # Update graphical illustration
            self.update_graph()

            self.update_critical_angle_plot()
            self.plot_beam_angle()
        except ValueError as e:
            QtWidgets.QMessageBox.critical(
                self, 'Error', str(e), QtWidgets.QMessageBox.StandardButton.Ok
            )
            return

    def update_table(self, wavelength, field_number):
        adapter_mag = self.camera_adapter_input.value()
        pixel_size = self.camera_pixel_size_input.value()

        self.table.setRowCount(0)  # Clear table
        data = [
            ['Manufacturer', self.objective.manufacturer.name],
            ['Magnification', f'{self.objective.magnification:.2f}'],
            ['Numerical Aperture', f'{self.objective.numerical_aperture:.2f}'],
            [
                'Immersion Medium',
                self.objective.immersion_medium.name + f' ({self.objective.n:.3f})',
            ],
            ['Focal Length (mm)', f'{self.objective.focal_length:.3f}'],
            ['Pupil Diameter (mm)', f'{self.objective.pupil_diameter:.3f}'],
            ['ParFocal Distance (mm)', f'{self.objective.parfocal_distance:.3f}'],
            [
                'Half Angle (degrees)',
                f'{np.degrees(self.objective.theta):.2f}',
            ],
            [
                'Critical Angle (degrees)',
                f'{np.degrees(self.objective.critical_angle):.2f}',
            ],
            [
                r'Collection Angle (% of hemisphere)',
                f'{self.objective.collection_angle:.2f}%',
            ],
            [
                'Field of View (um)',
                f'{self.objective.field_of_view(field_number, adapter_mag):.3f}',
            ],
        ]

        rayleigh_xy, rayleigh_z = self.objective.rayleigh_criterion(wavelength)
        optimal_pixel_size = self.objective.optimal_pixel_size(wavelength, adapter_mag)
        optimal_camera_adapter_mag = pixel_size / (
            self.objective.magnification * rayleigh_xy / 2000
        )
        data.extend(
            [
                [
                    f'Rayleigh Criterion XY @ {wavelength}nm (nm)',
                    f'{rayleigh_xy:.1f}',
                ],
                [
                    f'Rayleigh Criterion Z @ {wavelength}nm (nm)',
                    f'{rayleigh_z:.1f}',
                ],
                [
                    f'Nyquist Limit XY @ {wavelength}nm (nm)',
                    f'{self.objective.nyquist_limit(wavelength)[0]:.1f}',
                ],
                [
                    f'Optimal Pixel Size @ {wavelength}nm (um)',
                    f'{optimal_pixel_size:.3f}',
                ],
                [
                    f'Optimal Camera Adapter Magnification @ {wavelength}nm',
                    f'{optimal_camera_adapter_mag:.2f}',
                ],
            ]
        )

        # detector_size = self.objective.detector_size(field_number, wavelength)
        # data.append(
        #     [f'Detector Size @ {wavelength}nm (pixels)', f'{detector_size:.3f}']
        # )

        for row, (property_name, value) in enumerate(data):
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(property_name))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))

    def update_graph(self):
        self.plot_widget.clear()  # Clear previous plot
        _lambda = np.linspace(200, 1000, 100)
        xy_ray, z_ray = self.objective.rayleigh_criterion(_lambda)
        self.plot_widget.plot(
            _lambda,
            xy_ray,
            pen=pg.mkPen(color='b', width=2),
            name='XY Rayleigh Criterion',
        )
        self.plot_widget.plot(
            _lambda,
            z_ray,
            pen=pg.mkPen(color='r', width=2),
            name='Z Rayleigh Criterion',
        )

    def update_critical_angle_plot(self):
        self.critical_angle_plot.clear()
        # Get user inputs
        if (
            self.objective is not None
            and self.objective.immersion_medium != Immersion.AIR
            and self.objective.immersion_medium.value > Immersion.WATER.value
        ):
            angles, depths = TIRF.critical_angle(
                self.objective,
                n1=self.objective.immersion_medium.value,
                n2=Immersion.WATER.value,
            )

            for wavelength, depth in depths.items():
                self.critical_angle_plot.plot(
                    angles,
                    depth,
                    pen=pg.mkPen(color=self.COLORS[wavelength], width=1),
                    name=f'{wavelength}nm',
                )

            self.critical_angle_plot.addItem(
                pg.InfiniteLine(
                    (np.degrees(self.objective.critical_angle), 0),
                    angle=90,
                    label='Critical Angle',
                    labelOpts={'position': 0.1, 'color': 'y'},
                    pen=pg.mkPen(color='y', width=1, style=QtCore.Qt.PenStyle.DashLine),
                )
            )
            self.critical_angle_plot.addItem(
                pg.InfiniteLine(
                    (np.degrees(self.objective.theta), 0),
                    angle=90,
                    label='Max Angle',
                    labelOpts={'position': 0.15, 'color': 'r'},
                    pen=pg.mkPen(color='r', width=1, style=QtCore.Qt.PenStyle.DashLine),
                )
            )

            self.critical_angle_plot.setYRange(-25, 400)

    def plot_beam_angle(self):
        self.beam_angle.clear()
        if self.objective is not None:
            angles = np.arange(0, np.round(np.degrees(self.objective.theta), 1), 0.1)
            beam_positions = self.objective.beam_position(np.radians(angles))
            self.beam_angle.plot(
                beam_positions,
                angles,
                pen=pg.mkPen(color='g', width=2),
                name='Beam Position',
            )

    @classmethod
    def show_dialog(cls, parent: Optional[QtWidgets.QWidget] = None) -> None:
        '''Show the bridges manager widget as a dialog.'''
        if not hasattr(cls, '_singleton'):
            cls._singleton = cls(parent=parent)

        cls._singleton.exec()

    @classmethod
    def get_menu_action(cls, parent: Optional[QtWidgets.QWidget] = None) -> QAction:
        '''Get the action to show this widget in a menu.'''
        action = QAction('Objective Calculator', parent=parent)
        action.triggered.connect(lambda: cls.show_dialog(parent=parent))
        action.setStatusTip('Show the objective calculator')
        action.setToolTip('Show the objective calculator')
        return action


if __name__ == '__main__':
    # Example usage
    # obj = Objective(Manufacturer.NIKON, 60, 1.49, Immersion.OIL)

    # lambda_ = 550  # Wavelength in nm

    # obj.print(lambda_, field_number=200)
    # obj.print()  # Without wavelength

    app = QApplication(sys.argv)
    window = ObjectiveCalculator()
    window.show()
    sys.exit(app.exec())
