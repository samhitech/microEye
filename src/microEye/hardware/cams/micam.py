import ctypes
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
from tabulate import tabulate


class miCamera:
    def __init__(self, Cam_ID=0) -> None:
        self.Cam_ID = Cam_ID
        self.acquisition = False
        self.bytes_per_pixel = 1
        self.exposure_current = 1
        self.temperature = -127
        self.name = ''
        self.exposure_range = [0.05, 5000]
        self._width = 512
        self._height = 512

        self.status = {}

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    def get_temperature(self):
        return self.temperature

    def getWidth(self):
        if isinstance(self.width, ctypes.c_int):
            return self.width.value
        elif isinstance(self.width, int):
            return self.width

    def getHeight(self):
        if isinstance(self.height, ctypes.c_int):
            return self.height.value
        elif isinstance(self.height, int):
            return self.height

    def getExposure(self):
        if isinstance(self.exposure_current, ctypes.c_double):
            return self.exposure_current.value
        elif isinstance(self.exposure_current, (int, float)):
            return self.exposure_current

    def set_roi(self, x: int, y: int, width: int, height: int):
        '''
        Set the region of interest.

        This function sets the region of interest (ROI) for the camera.
        The ROI is defined by the top-left corner coordinates (x, y)
        and the width and height of the region.

        Parameters
        ----------
        x : int
            The x-coordinate of the top-left corner of the ROI.
        y : int
            The y-coordinate of the top-left corner of the ROI.
        width : int
            The width of the ROI.
        height : int
            The height of the ROI.

        Returns
        -------
        None
        '''
        pass

    def get_roi(self) -> tuple[int, int, int, int]:
        '''
        Return the current region of interest (ROI).

        Returns
        -------
        Tuple[int, int, int, int]
            The top-left corner coordinates (x, y) and the width and height of the ROI.
        '''
        pass

    def reset_roi(self):
        '''
        Reset the region of interest.
        '''
        pass


    def populate_status(self):
        pass

    def print_status(self):
        self.populate_status()
        for key in self.status:
            data = [[k, i] for k, i in self.status[key].items()]
            print(tabulate(data, headers=[key], tablefmt='rounded_grid'))

    def property_tree(self) -> list[dict[str, Any]]:
        '''
        Get the property tree for the camera.

        Returns
        -------
        list[dict[str, Any]]
            A dictionary containing the property tree for the camera.
        '''
        return []

    def update_cam(self, param, path: Optional[list[str]], param_value: Any):
        '''Update the camera parameters.

        Parameters
        ----------
        param : pyqtgraph.parametertree.Parameter
            The parameter to update.
        path : Optional[list[str]]
            The path to the parameter in tree.
        param_value : Any
            The new value for the parameter.
        '''
        pass

    def get_metadata(self) -> dict[str, Any]:
        '''
        Get the metadata for the camera.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the metadata for the camera.

        Example
        -------
        >>> return {
            'CHANNEL_NAME': 'Cam 0',
            'DET_MANUFACTURER': 'MicroEye',
            'DET_MODEL': 'Dummy',
            'DET_SERIAL': '123456789',
            'DET_TYPE': 'CMOS',
        }
        '''
        return {
            'CHANNEL_NAME': 'NA',
            'DET_MANUFACTURER': 'NA',
            'DET_MODEL': 'NA',
            'DET_SERIAL': 'NA',
            'DET_TYPE': 'NA',
        }

    @classmethod
    def get_camera_list(cls) -> list[dict[str, Any]]:
        '''
        Get a list of available cameras.

        Returns
        -------
        list[dict]
            A list of dictionaries containing camera information.
        '''
        return []


class DummyParams(Enum):
    FREERUN = 'Acquisition.Freerun'
    STOP = 'Acquisition.Stop'
    LOAD = 'Acquisition Settings.Load Config'
    SAVE = 'Acquisition Settings.Save Config'
    GAIN = 'Acquisition Settings.Gain'
    OFFSET = 'Acquisition Settings.Offset'
    HEIGHT = 'Acquisition Settings.Height'
    WIDTH = 'Acquisition Settings.Width'
    BINNING_HORIZONTAL = 'Acquisition Settings.Binning Horizontal'
    BINNING_VERTICAL = 'Acquisition Settings.Binning Vertical'
    BIT_DEPTH = 'Acquisition Settings.Bit Depth'
    FULL_WELL_CAPACITY = 'Acquisition Settings.Full Well Capacity'
    QUANTUM_EFFICIENCY = 'Acquisition Settings.Quantum Efficiency'
    DARK_CURRENT = 'Acquisition Settings.Dark Current'
    READOUT_NOISE = 'Acquisition Settings.Readout Noise'
    NOISE_BASELINE = 'Acquisition Settings.Noise Baseline'
    FLUX = 'Acquisition Settings.Flux'
    PATTERN_TYPE = 'Acquisition Settings.Pattern Type'
    PATTERN_OFFSET = 'Acquisition Settings.Pattern Offset'
    PATTERN_SINUSOIDAL = 'Acquisition Settings.Sinusoidal Pattern'
    SINUSOIDAL_FREQUENCY = 'Acquisition Settings.Sinusoidal Pattern.Frequency'
    SINUSOIDAL_PHASE = 'Acquisition Settings.Sinusoidal Pattern.Phase'
    SINUSOIDAL_AMPLITUDE = 'Acquisition Settings.Sinusoidal Pattern.Amplitude'
    SINUSOIDAL_DIRECTION = 'Acquisition Settings.Sinusoidal Pattern.Direction'

    DRIFT_SIM = 'Acquisition Settings.Drift Simulation'
    DRIFT_SPEED = DRIFT_SIM + '.Speed [pixel/frames]'
    DRIFT_PERIOD = DRIFT_SIM + '.Period [frames]'
    DRIFT_MOMENTUM = DRIFT_SIM + '.Momentum Coefficient'
    DRIFT_RANDOM_WALK = DRIFT_SIM + '.Random Walk Coefficient'
    DRIFT_VIBRATION = DRIFT_SIM + '.Vibration Amplitude'
    DRIFT_SIGMA_X = DRIFT_SIM + r'.$\sigma_x$ [pixels]'
    DRIFT_SIGMA_Y = DRIFT_SIM + r'.$\sigma_y$ [pixels]'
    DRIFT_AMPLITUDE = DRIFT_SIM + '.Amplitude'

    SM = 'Acquisition Settings.Single Molecule Sim'
    SM_INTENSITY = SM + '.Intensity [photons/loc]'
    SM_DENSITY = SM + '.Density [loc/um]'
    SM_WAVELENGTH = SM + '.Wavelength [nm]'
    SM_PIXEL_SIZE = SM + '.Projected Pixel Size [nm]'
    SM_NA = SM + '.Objective NA'

    def __str__(self):
        '''
        Return the last part of the enum value (Param name).
        '''
        return self.value.split('.')[-1]

    def get_path(self):
        '''
        Return the full parameter path.
        '''
        return self.value.split('.')


class miDummy(miCamera):
    instances = []

    def __init__(self, Cam_ID=0, **kwargs):
        super().__init__(Cam_ID)

        miDummy.instances.append(self)

        self.name = 'MicroEye Dummy Camera'
        self.exposure_current = kwargs.get('exposure', 10.0)
        self.exposure_unit = kwargs.get('exposure_unit', 'ms')
        self.exposure_increment = 1.0
        self.exposure_range = kwargs.get('exposure_range', [0.05, 5000])
        self.bytes_per_pixel = 2

        self._height = kwargs.get('height', 512)
        self._width = kwargs.get('width', 512)

        self.binning_horizontal = kwargs.get('binning_horizontal', 1)
        self.binning_vertical = kwargs.get('binning_vertical', 1)
        self.bit_depth = kwargs.get('bit_depth', 12)
        self.gain = kwargs.get('gain', 2.23)
        self.full_well_capacity = kwargs.get('full_well_capacity', 9200)
        self.quantum_efficiency = kwargs.get('quantum_efficiency', 0.8)
        self.dark_current = kwargs.get('dark_current', 0.0001)
        self.readout_noise = kwargs.get('readout_noise', 2.1)
        self.noise_baseline = kwargs.get('noise_baseline', 5)
        self.flux = kwargs.get('flux', 0)
        self.pattern_type = kwargs.get('pattern_type', 'Sinusoidal')
        self.pattern_offset = kwargs.get('pattern_offset', 0)

        # Sinusoidal pattern parameters
        self.sinusoidal_frequency = kwargs.get('sinusoidal_frequency', 0.01)
        self.sinusoidal_phase = kwargs.get('sinusoidal_phase', 0.0)
        self.sinusoidal_amplitude = kwargs.get('sinusoidal_amplitude', 0.1e5)
        self.sinusoidal_direction = kwargs.get('sinusoidal_direction', 'd')

        # Single molecule parameters
        self.sm_intensity = kwargs.get('sm_intensity', 5000)
        self.sm_density = kwargs.get('sm_density', 0.5)
        self.sm_pixel_size = kwargs.get('sm_pixel_size', 114.17)
        self.sm_wavelength = kwargs.get('sm_wavelength', 650)
        self.sm_na = kwargs.get('sm_na', 1.49)

        # Gaussian pattern Drift parameters
        self.drift_current_velocity = 0.0  # Current drift velocity
        self.drift_center_x = self.width // 2
        self.drift_center_y = self.height // 2
        self.drift_center_z = 0.0
        self.drift_fiducials = None
        self.drift_speed = 0.10
        self.drift_period = 80
        self.drift_momentum = 0.85
        self.drift_random_walk_factor = 0.80
        self.drift_vibration_amplitude = 0.1
        self.drift_sigma_x = 5
        self.drift_sigma_y = 20
        self.drift_amplitude = 5000

        self.__roi = None
        self._meshes = None

        self.update_meshes()

    def __del__(self):
        if self in miDummy.instances:
            miDummy.instances.remove(self)

    @property
    def exposure(self):
        return self.exposure_current

    @exposure.setter
    def exposure(self, value):
        if value < self.exposure_range[0]:
            value = self.exposure_range[0]
        elif value > self.exposure_range[1]:
            value = self.exposure_range[1]
        self.exposure_current = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if not isinstance(value, int) or value < 32 or value > 4096:
            raise ValueError('Height must be an integer between 32 and 4096.')
        self._height = value
        if self._width is not None:
            self._meshes = None
            self.update_meshes()
            if self.__roi:
                self.set_roi(*self.__roi)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if not isinstance(value, int) or value < 32 or value > 4096:
            raise ValueError('Width must be an integer between 32 and 4096.')
        self._width = value
        if self._height is not None:
            self._meshes = None
            self.update_meshes()
            if self.__roi:
                self.set_roi(*self.__roi)

    def update_meshes(self):
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        self._meshes = {
            'd': (X + Y),
            'h': X,
            'v': Y,
            'r': np.sqrt((X - self.width / 2) ** 2 + (Y - self.height / 2) ** 2),
        }

    def set_roi(self, x: int, y: int, width: int, height: int):
        detector_width = self.width
        detector_height = self.height

        if x == 0 and y == 0 and width == detector_width and height == detector_height:
            self.__roi = None
        else:
            x = max(0, min(x, detector_width - 1))
            y = max(0, min(y, detector_height - 1))
            width = max(1, min(width, detector_width - x))
            height = max(1, min(height, detector_height - y))
            self.__roi = (x, y, width, height)

    def get_roi(self) -> tuple[int, int, int, int]:
        if self.__roi is None:
            return 0, 0, self.width, self.height
        else:
            x, y, width, height = self.__roi
            return x, y, width, height

    def reset_roi(self):
        self.__roi = None

    def get_dummy_image(self, flux: Union[float, int, np.ndarray] = 10):
        '''
        Generate a dummy image with noise and CMOS conversion based on
        the camera properties.

        Parameters
        ----------
        flux : float | int | ndarray, optional
            The photon flux value used to generate a
            dummy image when no input image is provided [photons/s].
            Default is 10.

        Returns
        -------
        ndarray
            A 2D numpy array representing the dummy
            image with added noise and CMOS conversion.

        Notes
        -----
        The dummy image is generated based on the camera properties, including
        exposure time, dark current, quantum efficiency, full well capacity,
        readout noise, and gain.
        The exposure time, dark current, quantum efficiency, full well capacity,
        and gain are used to calculate the baseline, signal, and noise,
        which are then added to the generated dummy image. After that,
        the image is scaled based on the full well capacity and gain,
        clipped based on the bit depth, and cast to the desired data type.

        The exposure time is converted from milliseconds to seconds.
        A random noise image is generated with the desired width, height,
        and bit depth. The noise parameter now affects the scale, instead of
        the loc. The image is then scaled, clipped, and cast to the desired
        data type.

        '''
        if isinstance(flux, np.ndarray):
            if flux.shape != (self.height, self.width):
                raise ValueError(
                    f'Flux array must have shape ({self.height}, {self.width})!'
                )
        elif not isinstance(flux, float) and not isinstance(flux, int):
            raise TypeError('Flux must be a float, int or ndarray!')

        # Convert exposure time from milliseconds to seconds
        exposure_s = self.exposure_current / 1000

        baseline = self.dark_current * exposure_s

        signal = self.quantum_efficiency * flux * exposure_s

        # Generate a random noise image with the desired width, height, and bit depth
        # The noise parameter now affects the scale, instead of the loc
        image = np.random.poisson(
            lam=baseline + signal,  # mean
            size=(self.height, self.width),
        ) + np.random.normal(
            loc=0, scale=self.readout_noise, size=(self.height, self.width)
        )

        # Scale the image using quantum efficiency, full well capacity, and gain
        image = np.clip(image, 0, self.full_well_capacity) / self.gain

        # Add the baseline
        image += self.noise_baseline

        # Clip the image based on the bit depth
        image = np.clip(image, 0, 2**self.bit_depth - 1).astype(np.uint16)

        if self.__roi:
            x, y, width, height = self.__roi
            return image[y : y + height, x : x + width]
        else:
            return image

    def get_sinus_diagonal_pattern(
        self, time, amplitude=2000, frequency=0.001, phase=0, offset=0, type='d'
    ):
        '''
        Generate a sinusoidal diagonal pattern for the flux array.

        Parameters
        ----------
        time : float
            The current time.

        amplitude : float, optional
            The amplitude of the sinusoidal pattern (default is 1000).

        frequency : float, optional
            The frequency of the sinusoidal pattern (default is 0.01).

        phase : float, optional
            The phase of the sinusoidal pattern (default is 0).

        offset : float, optional
            The offset value to add to the pattern (default is 0).

        Returns
        -------
        ndarray
            A 2D numpy array representing the sinusoidal diagonal pattern.
        '''
        if type not in ['d', 'h', 'v', 'r']:
            type = 'd'

        pattern = (
            amplitude
            * np.sin(
                2 * np.pi * frequency * self._meshes[type] + phase + 2 * np.pi * time
            )
            + offset
        )

        if pattern.min() < 0:
            pattern += abs(pattern.min())

        return pattern

    def get_drift(self, time: float) -> float:
        '''
        Generate a random drift shift accounting for various factors.

        Parameters
        ----------
        time : float
            The current time.

        Returns
        -------
        float
            The amount of drift shift.
        '''

        # 1. Slow sinusoidal drift (thermal/mechanical creep)
        slow_drift = np.sin(2 * np.pi * time / self.drift_period)
        # 2. Brownian motion / random walk component
        # (use momentum to avoid jerky movement)
        random_component = np.random.normal(0, self.drift_random_walk_factor)
        self.drift_current_velocity = (
            self.drift_momentum * self.drift_current_velocity
            + (1 - self.drift_momentum) * random_component
        )
        # 3. Small high-frequency vibrations (building vibrations, etc)
        vibration = (
            self.drift_vibration_amplitude * np.sin(0.1 * time) * np.random.random()
        )
        # Combine all drift components
        drift_amount = (
            slow_drift * self.drift_speed + self.drift_current_velocity + vibration
        )

        return drift_amount

    def get_gaussian_beam_pattern(
        self,
        time: float,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
    ):
        '''
        Generate a Gaussian beam pattern for the flux array.
        The pattern is a 2D Gaussian distribution with a specified center and
        standard deviation.
        The pattern is also shifted in the x and y directions according to the
        drift speed.
        The pattern is also scaled by the amplitude.
        The pattern is also shifted in time according to the drift speed.

        Parameters
        ----------
        time : float
            The current time.
        amplitude : float, optional
            The amplitude of the Gaussian pattern (default is 5000).
        center_x : float, optional
            The x-coordinate of the center of the Gaussian pattern (default is None).
        center_y : float, optional
            The y-coordinate of the center of the Gaussian pattern (default is None).

        Returns
        -------
        ndarray
            A 2D numpy array representing the Gaussian beam pattern.
        '''
        if center_x is None:
            drift_amount = self.get_drift(time)

            # Update drift center with bounds checking
            self.drift_center_x += drift_amount
            edge_margin = max(10, self.width // 10)
            if self.drift_center_x < edge_margin:
                self.drift_center_x = edge_margin
                self.drift_current_velocity *= -0.5  # Bounce back with damping
            elif self.drift_center_x > self.width - edge_margin:
                self.drift_center_x = self.width - edge_margin
                self.drift_current_velocity *= -0.5  # Bounce back with damping

            center_x = self.drift_center_x
        if center_y is None:
            center_y = self.height // 2

        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        pattern = self.drift_amplitude * np.exp(
            -((X - center_x) ** 2) / (2 * self.drift_sigma_x**2)
            - ((Y - center_y) ** 2) / (2 * self.drift_sigma_y**2)
        )

        if pattern.min() < 0:
            pattern += abs(pattern.min())

        return pattern

    def get_astigmatic_fiducials_pattern(
        self,
        time: float,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
    ):
        '''
        Generate astigmatic fiducial markers for the flux array.

        It simulates axial drift by altering the sigma_x and sigma_y parameters.
        It simulates XY drift by altering the overall drift_center.

        Parameters
        ----------
        time : float
            The current time.
        center_x : float, optional
            The x-coordinate of the center of the fiducial markers
        center_y : float, optional
            The y-coordinate of the center of the fiducial markers
        center_z : float, optional
            The z-coordinate of the center of the fiducial markers

        Returns
        -------
        ndarray
            A 2D numpy array representing the astigmatic fiducial markers.
        '''

        drift = [self.get_drift(time) for _ in range(3)]

        if center_x is None:
            drift_amount = drift[0]

            # Update drift center with bounds checking
            self.drift_center_x += drift_amount * 0.1
            edge_margin = max(10, self.width // 10)
            if self.drift_center_x < edge_margin:
                self.drift_center_x = edge_margin
                self.drift_current_velocity *= -0.1  # Bounce back with damping
            elif self.drift_center_x > self.width - edge_margin:
                self.drift_center_x = self.width - edge_margin
                self.drift_current_velocity *= -0.1  # Bounce back with damping

            center_x = self.drift_center_x
        if center_y is None:
            drift_amount = drift[1]
            self.drift_center_y += drift_amount * 0.1
            edge_margin = max(10, self.height // 10)
            if self.drift_center_y < edge_margin:
                self.drift_center_y = edge_margin
                self.drift_current_velocity *= -0.1  # Bounce back with damping
            elif self.drift_center_y > self.height - edge_margin:
                self.drift_center_y = self.height - edge_margin
                self.drift_current_velocity *= -0.1  # Bounce back with damping
            center_y = self.drift_center_y
        if center_z is None:
            drift_amount = drift[2]
            self.drift_center_z += drift_amount * 10  # Z drift is faster

            if self.drift_center_z < -400:
                self.drift_center_z = -400
                self.drift_current_velocity *= -0.1  # Bounce back with damping
            elif self.drift_center_z > 400:
                self.drift_center_z = 400
                self.drift_current_velocity *= -0.1  # Bounce back with damping

            center_z = self.drift_center_z

        # convert center Z to sigma x and y
        sigma_0 = 3  # at Z = 0
        sigma_ratio_min = 0.25  # at Z min
        sigma_ratio_max = 4.0  # at Z max
        sigma_ratio = np.interp(
            self.drift_center_z, [-400, 0, 400], [sigma_ratio_min, 1, sigma_ratio_max]
        )  # sigma x / y
        sigma_x = sigma_0 if sigma_ratio >= 1 else sigma_0 * sigma_ratio
        sigma_y = sigma_0 if sigma_ratio <= 1 else sigma_0 / sigma_ratio

        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        if self.drift_fiducials is None:
            # generate random X Y points for the fiducials
            self.drift_fiducials = np.random.normal(
                loc=[center_x, center_y], scale=[50, 50], size=(5, 2)
            )

        pattern = self.drift_amplitude * np.exp(
            -((X - center_x) ** 2) / (2 * sigma_x**2)
            - ((Y - center_y) ** 2) / (2 * sigma_y**2)
        )

        for fid in self.drift_fiducials:
            x = fid[0]
            y = fid[1]

            pattern += self.drift_amplitude * np.exp(
                -((X - x) ** 2) / (2 * sigma_x**2) - ((Y - y) ** 2) / (2 * sigma_y**2)
            )

        if pattern.min() < 0:
            pattern += abs(pattern.min())

        return pattern

    def get_single_molecule_events(self):
        flux = np.zeros((self.height, self.width))

        pass

    def get_dummy_image_from_pattern(self, time):
        '''
        Generate a dummy image with noise and CMOS conversion based on
        the camera properties.

        Parameters
        ----------
        time : float
            The current time.

        Returns
        -------
        ndarray
            A 2D numpy array representing the dummy image
            with added noise and CMOS conversion.
        '''
        if self.pattern_type.lower() == 'sinusoidal':
            flux = self.get_sinus_diagonal_pattern(
                time,
                self.sinusoidal_amplitude,
                self.sinusoidal_frequency,
                self.sinusoidal_phase,
                self.pattern_offset,
                self.sinusoidal_direction,
            )
        elif self.pattern_type.lower() == 'single molecule sim':
            flux = 0
        elif self.pattern_type.lower() == 'gaussian':
            flux = self.get_gaussian_beam_pattern(time, center_x=None, center_y=None)
        elif self.pattern_type.lower() == 'astigmatic fiducials':
            flux = self.get_astigmatic_fiducials_pattern(time)
        else:
            flux = self.flux

        return self.get_dummy_image(flux)

    def get_metadata(self) -> dict[str, Any]:
        return {
            'CHANNEL_NAME': self.name,
            'DET_MANUFACTURER': 'MicroEye',
            'DET_MODEL': 'Dummy',
            'DET_SERIAL': '123456789',
            'DET_TYPE': 'CMOS',
        }

    def property_tree(self):
        HEIGHT = {
            'name': str(DummyParams.HEIGHT),
            'type': 'int',
            'value': 512,
            'limits': [0, 4096],
        }
        WIDTH = {
            'name': str(DummyParams.WIDTH),
            'type': 'int',
            'value': 512,
            'limits': [0, 4096],
        }
        BINNING_HORIZONTAL = {
            'name': str(DummyParams.BINNING_HORIZONTAL),
            'type': 'int',
            'value': 1,
            'limits': [0, 10],
        }
        BINNING_VERTICAL = {
            'name': str(DummyParams.BINNING_VERTICAL),
            'type': 'int',
            'value': 1,
            'limits': [0, 10],
        }
        BIT_DEPTH = {
            'name': str(DummyParams.BIT_DEPTH),
            'type': 'list',
            'limits': [8, 10, 12, 16],
            'value': 12,
        }
        GAIN = {'name': str(DummyParams.GAIN), 'type': 'float', 'value': 2.23}
        FULL_WELL_CAPACITY = {
            'name': str(DummyParams.FULL_WELL_CAPACITY),
            'type': 'int',
            'value': 9200,
            'suffix': 'e-',
        }
        QUANTUM_EFFICIENCY = {
            'name': str(DummyParams.QUANTUM_EFFICIENCY),
            'type': 'float',
            'value': 0.8,
            'limits': [0, 1],
        }
        DARK_CURRENT = {
            'name': str(DummyParams.DARK_CURRENT),
            'type': 'float',
            'value': 0.0001,
            'suffix': ' e-/s',
        }
        READOUT_NOISE = {
            'name': str(DummyParams.READOUT_NOISE),
            'type': 'float',
            'value': 2.1,
            'suffix': ' e-',
        }
        NOISE_BASELINE = {
            'name': str(DummyParams.NOISE_BASELINE),
            'type': 'float',
            'value': 5.0,
            'limits': [0, 2**16],
            'suffix': ' ADU',
        }
        FLUX = {
            'name': str(DummyParams.FLUX),
            'type': 'float',
            'value': 0,
            'suffix': ' e-/p/s',
        }
        PATTERN_TYPE = {
            'name': str(DummyParams.PATTERN_TYPE),
            'type': 'list',
            'limits': [
                'Constant Flux',
                'Sinusoidal',
                'Gaussian',
                'Astigmatic Fiducials',
                'Single Molecule Sim',
            ],
            'value': 'Sinusoidal',
        }
        PATTERN_OFFSET = {
            'name': str(DummyParams.PATTERN_OFFSET),
            'type': 'float',
            'value': 0.0,
            'suffix': ' e-',
        }
        PATTERN_SINUSOIDAL = {
            'name': str(DummyParams.PATTERN_SINUSOIDAL),
            'type': 'group',
            'expanded': False,
            'children': [
                {
                    'name': str(DummyParams.SINUSOIDAL_FREQUENCY),
                    'type': 'float',
                    'value': 0.01,
                    'suffix': ' Hz',
                },
                {
                    'name': str(DummyParams.SINUSOIDAL_PHASE),
                    'type': 'float',
                    'value': 0.0,
                    'suffix': ' deg',
                },
                {
                    'name': str(DummyParams.SINUSOIDAL_AMPLITUDE),
                    'type': 'float',
                    'value': 0.1e5,
                    'suffix': ' e-/p/s',
                },
                {
                    'name': str(DummyParams.SINUSOIDAL_DIRECTION),
                    'type': 'list',
                    'limits': ['d', 'h', 'v', 'r'],
                    'value': 'd',
                },
            ],
        }
        SM_SIM = {
            'name': str(DummyParams.SM),
            'type': 'group',
            'expanded': False,
            'children': [
                {
                    'name': str(DummyParams.SM_INTENSITY),
                    'type': 'float',
                    'value': 5000,
                    'decimals': 6,
                },
                {
                    'name': str(DummyParams.SM_DENSITY),
                    'type': 'float',
                    'value': 0.5,
                    'decimals': 6,
                },
                {
                    'name': str(DummyParams.SM_PIXEL_SIZE),
                    'type': 'float',
                    'value': 114.17,
                    'decimals': 6,
                },
                {
                    'name': str(DummyParams.SM_WAVELENGTH),
                    'type': 'float',
                    'value': 650,
                    'decimals': 6,
                },
                {
                    'name': str(DummyParams.SM_NA),
                    'type': 'float',
                    'value': 1.49,
                    'decimals': 6,
                },
            ],
        }

        # drift simulation
        DRIFT_SIM = {
            'name': str(DummyParams.DRIFT_SIM),
            'type': 'group',
            'expanded': False,
            'children': [
                {
                    'name': str(DummyParams.DRIFT_SPEED),
                    'type': 'float',
                    'value': 0.1,
                },
                {
                    'name': str(DummyParams.DRIFT_PERIOD),
                    'type': 'int',
                    'value': 90,
                },
                {
                    'name': str(DummyParams.DRIFT_MOMENTUM),
                    'type': 'float',
                    'value': 0.8,
                    'limits': [0, 1],
                },
                {
                    'name': str(DummyParams.DRIFT_RANDOM_WALK),
                    'type': 'float',
                    'value': 0.80,
                },
                {
                    'name': str(DummyParams.DRIFT_VIBRATION),
                    'type': 'float',
                    'value': 0.1,
                },
                {
                    'name': str(DummyParams.DRIFT_SIGMA_X),
                    'type': 'float',
                    'value': 5,
                },
                {
                    'name': str(DummyParams.DRIFT_SIGMA_Y),
                    'type': 'float',
                    'value': 20,
                },
                {
                    'name': str(DummyParams.DRIFT_AMPLITUDE),
                    'type': 'float',
                    'value': 5000,
                }
            ],
        }

        return [
            HEIGHT,
            WIDTH,
            BINNING_HORIZONTAL,
            BINNING_VERTICAL,
            BIT_DEPTH,
            GAIN,
            FULL_WELL_CAPACITY,
            QUANTUM_EFFICIENCY,
            DARK_CURRENT,
            READOUT_NOISE,
            NOISE_BASELINE,
            FLUX,
            PATTERN_TYPE,
            PATTERN_OFFSET,
            PATTERN_SINUSOIDAL,
            DRIFT_SIM,
            SM_SIM
        ]

    def update_cam(self, param, path, param_value):
        if path is None:
            return

        param_value = param.value()

        try:
            param_name = DummyParams('.'.join(path))
        except ValueError:
            return

        if param_name == DummyParams.HEIGHT:
            self.height = param_value
        elif param_name == DummyParams.WIDTH:
            self.width = param_value
        elif param_name == DummyParams.BINNING_HORIZONTAL:
            self.binning_horizontal = param_value
        elif param_name == DummyParams.BINNING_VERTICAL:
            self.binning_vertical = param_value
        elif param_name == DummyParams.BIT_DEPTH:
            self.bit_depth = param_value
        elif param_name == DummyParams.GAIN:
            self.gain = param_value
        elif param_name == DummyParams.FULL_WELL_CAPACITY:
            self.full_well_capacity = param_value
        elif param_name == DummyParams.QUANTUM_EFFICIENCY:
            self.quantum_efficiency = param_value
        elif param_name == DummyParams.DARK_CURRENT:
            self.dark_current = param_value
        elif param_name == DummyParams.READOUT_NOISE:
            self.readout_noise = param_value
        elif param_name == DummyParams.NOISE_BASELINE:
            self.noise_baseline = param_value
        elif param_name == DummyParams.FLUX:
            self.flux = param_value
        elif param_name == DummyParams.PATTERN_TYPE:
            self.pattern_type = param_value
        elif param_name == DummyParams.PATTERN_OFFSET:
            self.pattern_offset = param_value

        elif param_name == DummyParams.SINUSOIDAL_AMPLITUDE:
            self.sinusoidal_amplitude = param_value
        elif param_name == DummyParams.SINUSOIDAL_FREQUENCY:
            self.sinusoidal_frequency = param_value
        elif param_name == DummyParams.SINUSOIDAL_PHASE:
            self.sinusoidal_phase = param_value
        elif param_name == DummyParams.SINUSOIDAL_DIRECTION:
            self.sinusoidal_direction = param_value

        elif param_name == DummyParams.SM_INTENSITY:
            self.sm_intensity = param_value
        elif param_name == DummyParams.SM_DENSITY:
            self.sm_density = param_value
        elif param_name == DummyParams.SM_PIXEL_SIZE:
            self.sm_pixel_size = param_value
        elif param_name == DummyParams.SM_WAVELENGTH:
            self.sm_wavelength = param_value
        elif param_name == DummyParams.SM_NA:
            self.sm_na = param_value

        elif param_name == DummyParams.DRIFT_SPEED:
            self.drift_speed = param_value
        elif param_name == DummyParams.DRIFT_PERIOD:
            self.drift_period = param_value
        elif param_name == DummyParams.DRIFT_MOMENTUM:
            self.drift_momentum = param_value
        elif param_name == DummyParams.DRIFT_RANDOM_WALK:
            self.drift_random_walk_factor = param_value
        elif param_name == DummyParams.DRIFT_VIBRATION:
            self.drift_vibration_amplitude = param_value
        elif param_name == DummyParams.DRIFT_SIGMA_X:
            self.drift_sigma_x = param_value
        elif param_name == DummyParams.DRIFT_SIGMA_Y:
            self.drift_sigma_y = param_value
        elif param_name == DummyParams.DRIFT_AMPLITUDE:
            self.drift_amplitude = param_value

    @classmethod
    def get_camera_list(cls):
        return [
            {
                'Camera ID': 'Cam 0',
                'Device ID': 0,
                'Model': 'Dummy',
                'Serial': '42023060',
                'InUse': bool(miDummy.instances),
                'Status': 'Ready',
                'Sensor ID': '06032024',
                'Driver': 'miDummy',
            }
        ]


if __name__ == '__main__':
    import timeit

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    # create a dummy camera object
    cam = miDummy()
    cam.pattern_type = 'gaussian'

    # set up animation
    fig, ax = plt.subplots()
    im = ax.imshow(cam.get_dummy_image_from_pattern(0), cmap='gray', vmin=0)

    plt.colorbar(im)

    def update(frame):
        img = cam.get_dummy_image_from_pattern(frame)
        im.set_data(img)
        return (im,)

    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(0, 5, 0.05), interval=1, blit=True
    )
    # ani.save('animation_1.gif', writer='pillow', fps=30)
    plt.show()
