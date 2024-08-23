from queue import Queue
from typing import Union

import numpy as np

from microEye.hardware.pycromanager.core import PycroCore
from microEye.hardware.pycromanager.enums import DeviceType


class PycroDevice:
    def __init__(self, label: str, type: DeviceType) -> None:
        self._core = PycroCore.instance()

        if not isinstance(type, DeviceType):
            raise ValueError(f'Invalid device type: {type}')

        if label is None or not isinstance(label, str):
            if type == DeviceType.CameraDevice:
                label = self._core.get_camera_device()
            elif type == DeviceType.StageDevice:
                label = self._core.get_focus_device()
            elif type == DeviceType.XYStageDevice:
                label = self._core.get_xy_stage_device()
            elif type == DeviceType.AutoFocusDevice:
                label = self._core.get_auto_focus_device()
            elif type == DeviceType.GalvoDevice:
                label = self._core.get_galvo_device()
            elif type == DeviceType.SLMDevice:
                label = self._core.get_slm_device()
            elif type == DeviceType.ShutterDevice:
                label = self._core.get_shutter_device()
            else:
                raise ValueError(
                    f'Unsupported device type: {type}. Please provide a valid label.'
                )

        if self._core.get_device_type(label) != type:
            raise ValueError(f'Label "{label}" is not a {type.name} device')

        self._label = label
        self._type = type

    @property
    def device_type(self) -> DeviceType:
        return self._type

    @property
    def label(self) -> str:
        return self._label

    @property
    def busy(self) -> bool:
        return self._core.device_busy(self._label)

    @property
    def name(self) -> float:
        return self._core.get_device_name(self._label)

    def wait_for_device(self):
        self._core.wait_for_device(self._label)

    def get_prop_names(self) -> list[str]:
        return self._core.get_device_property_names(self._label)

    def get_prop_values(self) -> dict:
        return self._core.get_system_state()[self._label]

    def get_property(self, prop_name: str):
        return self._core.get_property(self.label, prop_name)

    def set_property(self, prop_name: str, prop_value):
        self._core.set_property(self.label, prop_name, prop_value)

    def __str__(self) -> str:
        return (
            f'Device Type: {self.device_type.name}, '
            f'Label: {self.label}, Name: {self.name}'
        )


class PycroCamera(PycroDevice):
    def __init__(self, label: str = None, buffer: Queue = None) -> None:
        super().__init__(label, DeviceType.CameraDevice)
        self._core = PycroCore.instance()
        self._last_image = np.zeros((self.height, self.width))
        self._buffer = buffer if buffer is not None else Queue()

    @property
    def width(self) -> int:
        return self._core.get_image_width()

    @property
    def height(self) -> int:
        return self._core.get_image_height()

    @property
    def ROI(self) -> int:
        roi = self._core.get_roi()
        return roi.x, roi.y, roi.width, roi.height

    def set_roi(self, roi_x, roi_y, roi_width, roi_height):
        self._core.set_roi(roi_x, roi_y, roi_width, roi_height, self.label)

    @property
    def bit_depth(self) -> int:
        return self._core.get_image_bit_depth()

    @property
    def exposure(self) -> float:
        return self._core.get_exposure()

    @exposure.setter
    def exposure(self, value: float):
        return self._core.set_exposure(value)

    @property
    def pixel_size_um(self) -> float:
        return self._core.get_pixel_size_um()

    def get_property(self, prop_name: str):
        return self._core.get_property(self.label, prop_name)

    def set_property(self, prop_name: str, prop_value):
        self._core.set_property(self.label, prop_name, prop_value)

    def get_image(self) -> np.ndarray:
        return self._core.get_image().reshape((cam.height, cam.width))

    def get_tagged_image(self) -> tuple[np.ndarray, dict[str, str]]:
        tagged = self._core.get_tagged_image()
        return tagged.pix, tagged.tags

    def snap_image(self, buffered=True):
        self._core.snap_image()
        img: np.ndarray = self.get_image()
        if self._last_image is None or not np.allclose(self._last_image, img):
            self._last_image = img.copy()
            if buffered:
                self._buffer.put(img)
            else:
                return img

    def set_as_camera_device(self):
        self._core.set_camera_device(self.label)


class PycroStage(PycroDevice):
    def __init__(self, label: str = None) -> None:
        super().__init__(label, DeviceType.StageDevice)

    @property
    def position(self) -> float:
        return self._core.get_position(self.label)

    @position.setter
    def position(self, value: float):
        self._core.set_position(value, self.label)

    def move_rel(self, step: float):
        self._core.set_relative_position(step, self.label)

    def move_abs(self, position: float):
        self._core.set_position(position, self.label)

    def home(self):
        return self._core.home(self.label)

    def stop(self):
        return self._core.stop(self.label)

    def set_adapter_origin_z(self, new_z_um: float):
        self._core.set_adapter_origin(new_z_um, self.label)

    def set_origin(self):
        self._core.set_origin(self.label)


class PycroXYStage(PycroDevice):
    def __init__(self, label: str = None) -> None:
        super().__init__(label, DeviceType.XYStageDevice)

    @property
    def X(self) -> float:
        return self._core.get_x_position(self.label)

    @property
    def Y(self) -> float:
        return self._core.get_y_position(self.label)

    @X.setter
    def X(self, value: float):
        self._core.set_xy_position(value, self.Y, self.label)

    @Y.setter
    def Y(self, value: float):
        self._core.set_xy_position(self.X, value, self.label)

    def move_rel(self, x: float, y: float):
        self._core.set_relative_xy_position(x, y, self.label)

    def move_abs(self, x: float, y: float):
        self._core.set_xy_position(x, y, self.label)

    def home(self):
        return self._core.home(self.label)

    def stop(self):
        return self._core.stop(self.label)

    def set_adapter_origin(self, new_x_um: float, new_y_um: float):
        self._core.set_adapter_origin_xy(new_x_um, new_y_um, self.label)

    def set_origin(self):
        self._core.set_origin_xy(self.label)

    def set_as_xy_stage_device(self):
        self._core.set_xy_stage_device(self.label)


if __name__ == '__main__':
    import cv2

    cam = PycroCamera()
    cam.exposure = 100

    stage = PycroStage(cam._core)

    print(stage.stop_z())

    while True:
        cam.snap_image()
        img = cam._buffer.get()

        cv2.imshow('', img)
        cv2.waitKey(1000)
