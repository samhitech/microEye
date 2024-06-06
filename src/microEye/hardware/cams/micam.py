import ctypes
from typing import Union

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

        self.status = {}

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

    def populate_status(self):
        pass

    def print_status(self):
        self.populate_status()
        for key in self.status:
            data = [[k, i] for k, i in self.status[key].items()]
            print(tabulate(data, headers=[key], tablefmt='rounded_grid'))


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
        self.sinusoidal_frequency = kwargs.get('sinusoidal_frequency', 0.01)
        self.sinusoidal_phase = kwargs.get('sinusoidal_phase', 0.0)
        self.sinusoidal_amplitude = kwargs.get('sinusoidal_amplitude', 0.1e5)
        self.sinusoidal_direction = kwargs.get('sinusoidal_direction', 'd')

        self.sm_intensity = kwargs.get('sm_intensity', 5000)
        self.sm_density = kwargs.get('sm_density', 0.5)
        self.sm_pixel_size = kwargs.get('sm_pixel_size', 114.17)
        self.sm_wavelength = kwargs.get('sm_wavelength', 650)
        self.sm_na = kwargs.get('sm_na', 1.49)

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
            'd' : (X + Y),
            'h' : X,
            'v' : Y,
            'r' : np.sqrt(
                (X - self.width/2)**2 + (Y - self.height/2)**2)
        }

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
        '''
        Return the current region of interest (ROI).

        Returns
        -------
        Tuple[int, int, int, int]
            The top-left corner coordinates (x, y) and the width and height of the ROI.
        '''
        if self.__roi is None:
            return 0, 0, self.width, self.height
        else:
            x, y, width, height = self.__roi
            return x, y, width, height

    def reset_roi(self):
        '''
        Reset the region of interest.
        '''
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
                    f'Flux array must have shape ({self.height}, {self.width})!')
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
            size=(self.height, self.width)
            ) + np.random.normal(
            loc=0,
            scale=self.readout_noise,
            size=(self.height, self.width))

        # Scale the image using quantum efficiency, full well capacity, and gain
        image = np.clip(image, 0, self.full_well_capacity) / self.gain

        # Add the baseline
        image += self.noise_baseline

        # Clip the image based on the bit depth
        image = np.clip(image, 0, 2**self.bit_depth - 1).astype(np.uint16)

        if self.__roi:
            x, y, width, height = self.__roi
            return image[y:y+height, x:x+width]
        else:
            return image

    def get_sinus_diagonal_pattern(
            self, time, amplitude=2000, frequency=0.001, phase=0, offset=0, type='d'):
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

        pattern = amplitude * np.sin(
            2 * np.pi * frequency * self._meshes[type] + phase + 2 * np.pi * time
            ) + offset

        if pattern.min() < 0:
            pattern += abs(pattern.min())

        return pattern

    def get_single_molecule_events(self):
        flux = np.zeros((self.height, self.width))


        pass

    def get_dummy_image_from_pattern(
            self, time):
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
                time, self.sinusoidal_amplitude,
                self.sinusoidal_frequency,
                self.sinusoidal_phase,
                self.pattern_offset,
                self.sinusoidal_direction)
        elif self.pattern_type.lower() == 'single molecule sim':
            flux = 0
        else:
            flux = self.flux

        return self.get_dummy_image(flux)

    @staticmethod
    def get_camera_list():
        return [
            {
                'Camera ID': 'Cam 0',
                'Device ID': 0, 'Model': 'Dummy',
                'Serial': '42023060', 'InUse': bool(miDummy.instances),
                'Status': 'Ready',
                'Sensor ID': '06032024', 'Driver': 'miDummy'}]

if __name__ == '__main__':
    import timeit

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

#     setup = '''
# import numpy as np
# from micam import miCamera, miDummy

# cam = miDummy()
# type = 'r'
#     '''

#     stmt = '''
# cam.get_dummy_image_from_pattern(0, offset=1000, type=type)
#     '''

#     total_time = timeit.timeit(stmt=stmt, setup=setup, number=100)
#     avg_time = total_time / 100
#     print(f'Average time per call: {avg_time:.6f} seconds')

    # create a dummy camera object
    cam = miDummy()
    type = 'r'

    # set up animation
    fig, ax = plt.subplots()
    im = ax.imshow(
        cam.get_dummy_image_from_pattern(
            0, offset=1000, type=type), cmap='gray', vmin=0)

    plt.colorbar(im)

    def update(frame):
        img = cam.get_dummy_image_from_pattern(frame, offset=1000, type=type)
        im.set_data(img)
        return im,

    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(0, 5, 0.05), interval=1, blit=True)
    # ani.save('animation_1.gif', writer='pillow', fps=30)
    plt.show()

