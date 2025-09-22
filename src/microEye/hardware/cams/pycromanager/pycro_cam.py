import time

import numpy as np

from microEye.hardware.cams.micam import miCamera
from microEye.hardware.pycromanager.core import DEFAULT_BRIDGE_PORT, PycroCore
from microEye.hardware.pycromanager.devices import PycroDevice
from microEye.hardware.pycromanager.enums import DeviceType


class PycroCamera(PycroDevice, miCamera):
    def __init__(self, label: str = None, port: int = DEFAULT_BRIDGE_PORT) -> None:
        PycroDevice.__init__(self, label, DeviceType.CameraDevice, port)
        miCamera.__init__(self)
        # self._last_image = np.zeros((self.height, self.width))
        # self._buffer = buffer if buffer is not None else Queue()

    @property
    def image_count(self) -> int:
        return self._core.get_remaining_image_count()

    @property
    def width(self) -> int:
        return self._core.get_image_width()

    @width.setter
    def width(self, value: int):
        raise NotImplementedError('Cannot set camera width')

    @property
    def height(self) -> int:
        return self._core.get_image_height()

    @height.setter
    def height(self, value: int):
        raise NotImplementedError('Cannot set camera height')

    @property
    def ROI(self) -> tuple[int, int, int, int]:
        return self._core.get_roi()

    def get_roi(self) -> tuple[int, int, int, int]:
        return self.ROI

    def set_roi(self, roi_x, roi_y, roi_width, roi_height):
        self._core.set_roi(roi_x, roi_y, roi_width, roi_height, self.label)

    def reset_roi(self):
        self._core.clear_roi()

    @property
    def bit_depth(self) -> int:
        return self._core.get_image_bit_depth()

    @property
    def bytes_per_pixel(self) -> int:
        return int(np.ceil(self.bit_depth / 8))

    @bytes_per_pixel.setter
    def bytes_per_pixel(self, value: int):
        pass

    @property
    def exposure_current(self) -> float:
        return self._core.get_exposure(self._label)

    @exposure_current.setter
    def exposure_current(self, value: float):
        self._core.set_exposure(value, self.label)

    def setExposure(self, exposure: float):
        self._core.set_exposure(exposure, self.label)

    @property
    def exposure_range(self) -> tuple[float, float]:
        try:
            return self.get_property_limits('Exposure')
        except Exception as e:
            return (0.1, 10000)

    @exposure_range.setter
    def exposure_range(self, value: tuple[float, float]):
        return
        print('PycroCamera: Setting exposure range is not supported')

    def get_temperature(self) -> float:
        try:
            for prop in self.get_prop_names():
                if 'temperature' in prop.lower():
                    return float(self.get_property(prop))
        except Exception as e:
            print(f'Error getting temperature: {e}')
        finally:
            return -127.0  # noqa: B012

    @property
    def pixel_size_um(self) -> float:
        return self._core.get_pixel_size_um()

    def get_property(self, prop_name: str):
        return self._core.get_property(self.label, prop_name)

    def set_property(self, prop_name: str, prop_value):
        self._core.set_property(self.label, prop_name, prop_value)

    def get_image(self) -> np.ndarray:
        return self._core.get_image().reshape((self.height, self.width))

    # def get_image_buffer(self) -> np.ndarray:
    #     return self._buffer.get().reshape((cam.height, cam.width))

    def pop_image(self) -> np.ndarray:
        return self._core.pop_next_image().reshape((self.height, self.width))

    def get_tagged_image(self) -> tuple[np.ndarray, dict[str, str]]:
        tagged = self._core.get_tagged_image()
        return tagged.pix, tagged.tags

    def snap_image(self, buffered=False):
        if self.is_sequence_running():
            print('Sequence already running')
            return None

        self.start_sequence_acquisition(0, 1)

        # add timeout to avoid infinite loop
        start = time.time()
        while self.image_count == 0:
            time.sleep(0.001)

            if time.time() - start > 5:
                print('Timeout: Could not snap image')
                return None

        img: np.ndarray = self.pop_image()
        # if self._last_image is None or not np.allclose(self._last_image, img):
        #     self._last_image = img.copy()
        #     if buffered:
        #         self._buffer.put(img)

        return img

    def start_continuous_acquisition(self, interval_ms: int = 0):
        if self._core.is_sequence_running():
            print('Sequence already running')
            return

        self._core.start_continuous_sequence_acquisition(interval_ms)

    def is_sequence_running(self) -> bool:
        return self._core.is_sequence_running(self._label)

    def start_sequence_acquisition(self, interval_ms: float, num_time_points: int):
        if self.is_sequence_running():
            print('Sequence already running')
            return

        self._core.prepare_sequence_acquisition(self._label)
        self._core.start_sequence_acquisition(
            num_time_points, interval_ms, True, self._label
        )

    def stop_sequence_acquisition(self):
        self._core.stop_sequence_acquisition(self._label)

    def set_as_camera_device(self):
        self._core.set_camera_device(self._label)

    def populate_status(self):
        self.status['Camera'] = {
            'Class': self.__class__.__name__,
            'Name': self._label,
            'Type': self._type.name,
        }

        self.status['Exposure'] = {
            'Value': self.exposure_current,
            'Unit': 'ms',
        }

    def get_metadata(self):
        return {
            'CHANNEL_NAME': self.name,
            'DET_MANUFACTURER': 'Pycromanager',
            'DET_MODEL': self.label,
            'DET_SERIAL': 'N/A',
            'DET_TYPE': 'CMOS',
        }

    @classmethod
    def get_camera_list(cls):
        return (
            PycroCore.get_camera_list() if hasattr(PycroCore, 'get_camera_list') else []
        )
