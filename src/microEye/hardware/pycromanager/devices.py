import time

import numpy as np

from microEye.hardware.cams.micam import miCamera
from microEye.hardware.pycromanager.core import DEFAULT_BRIDGE_PORT, PycroCore
from microEye.hardware.pycromanager.enums import DeviceType, PropertyType
from microEye.hardware.stages.stage import AbstractStage, Axis, Units


class PycroDevice:
    def __init__(
        self, label: str, type: DeviceType, port: int = DEFAULT_BRIDGE_PORT
    ) -> None:
        if not isinstance(type, DeviceType):
            raise ValueError(f'Invalid device type: {type}')

        self._port = port

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
    def _core(self) -> PycroCore:
        return PycroCore.instance(self._port)

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

    @name.setter
    def name(self, value: str):
        return
        print('Setting PycroDevice name is not supported')

    def wait_for_device(self):
        self._core.wait_for_device(self._label)

    def get_prop_names(self) -> list[str]:
        return self._core.get_device_property_names(self._label)

    def get_prop_values(self) -> dict:
        return self._core.get_system_state()[self._label]

    def has_property(self, prop_name: str) -> bool:
        return self._core.has_property(self._label, prop_name)

    def get_property(self, prop_name: str):
        prop_value = self._core.get_property(self.label, prop_name)
        prop_type = self.get_property_type(prop_name)

        if prop_type == PropertyType.Undef:
            return prop_value

        python_type = prop_type.to_python()

        return python_type(prop_value)

    def get_property_type(self, prop_name: str) -> PropertyType:
        return self._core.get_property_type(self.label, prop_name)

    def set_property(self, prop_name: str, prop_value):
        self._core.set_property(self.label, prop_name, prop_value)

    def is_property_read_only(self, prop_name: str) -> bool:
        return self._core.is_property_read_only(self.label, prop_name)

    def get_allowed_property_values(self, prop_name: str) -> list:
        return self._core.get_allowed_property_values(self.label, prop_name)

    def get_property_limits(self, prop_name: str) -> tuple:
        return self._core.get_property_limits(self.label, prop_name)

    def property_tree(self):
        tree = []
        for prop in self.get_prop_names():
            prop_type = self.get_property_type(prop)

            if prop_type == PropertyType.Undef:
                continue

            python_type = prop_type.to_python()

            prop_value = python_type(self.get_property(prop))

            prop_read_only = self.is_property_read_only(prop)

            prop_limits = self.get_property_limits(prop)

            prop_allowed_values = self.get_allowed_property_values(prop)

            # if any prop_limits is None, it means that the property has no limits
            if any(limit is None for limit in prop_limits):
                prop_limits = None

            if prop_allowed_values:
                prop_limits = [python_type(value) for value in prop_allowed_values]
                python_type = list

            tree.append(
                {
                    'name': prop,
                    'type': python_type.__name__,
                    'value': prop_value,
                    'enabled': not prop_read_only,
                    'limits': prop_limits,
                }
            )

        return tree

    def __str__(self) -> str:
        return (
            f'Device Type: {self.device_type.name}, '
            f'Label: {self.label}, Name: {self.name}'
        )


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


class PycroStage(PycroDevice, AbstractStage):
    def __init__(self, label: str = None, port: int = DEFAULT_BRIDGE_PORT) -> None:
        PycroDevice.__init__(self, label, DeviceType.StageDevice, port=port)

        AbstractStage.__init__(
            self, f'PycroStage {label}', max_range=100, units=Units.MICROMETERS
        )

    def is_open(self) -> bool:
        return False

    def connect(self):
        pass

    def disconnect(self):
        pass

    @property
    def position(self) -> int:
        pos = self.convert_to_nm(self._core.get_position(self.label), axis=Axis.Z)
        self.set_position(axis=Axis.Z, position=pos)
        return pos

    @position.setter
    def position(self, value: int):
        self._core.set_position(self.convert_from_nm(value, axis=Axis.Z), self.label)
        self.set_position(axis=Axis.Z, position=value)

    def move_relative(self, step: float):
        self._core.set_relative_position(
            self.convert_from_nm(step, axis=Axis.Z), self.label
        )

        self.set_position(axis=Axis.Z, position=step, incremental=True)

    def move_absolute(self, position: float):
        self._core.set_position(self.convert_from_nm(position, axis=Axis.Z), self.label)
        self.set_position(axis=Axis.Z, position=position)

    def home(self):
        # self._core.home(self.label)
        self.move_absolute(0)
        return self.position

    def stop(self):
        self._core.stop(self.label)
        return self.position

    def set_adapter_origin_z(self, new_z_um: float):
        self._core.set_adapter_origin(new_z_um, self.label)

    def set_origin(self):
        self._core.set_origin(self.label)

    def refresh_position(self):
        self.move_absolute(self.position)


class PycroXYStage(PycroDevice):
    def __init__(self, label: str = None, port: int = DEFAULT_BRIDGE_PORT) -> None:
        super().__init__(label, DeviceType.XYStageDevice, port=port)

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
    import os
    import time

    import cv2

    mm_path = 'C:/Program Files/Micro-Manager-2.0'
    config = 'MMConfig_demo.cfg'
    port = DEFAULT_BRIDGE_PORT

    try:
        # start_headless(
        #     mm_path,
        #     config,
        #     port=port,
        #     # python_backend=True,
        # )

        # stage = PycroStage(port=port)
        # print(stage)
        # print(stage.position)
        # stage.move_absolute(50)
        # print(stage.position)

        cam = PycroCamera('Camera', port=port)
        cam.exposure_current = 50

        cam.start_sequence_acquisition(0, 500)  # 0 ms interval for fastest acquisition

        # def show_image(image, metadata):
        #     cv2.imshow('Image', image)
        #     cv2.waitKey(1)

        # with Acquisition(show_display=False, image_process_fn=show_image) as acq:
        #     events = multi_d_acquisition_events(
        #         2000,
        #     )  # events for 2000 time points
        #     for dic in events:
        #         dic['camera'] = 'DCam_1'
        #     acq.acquire(events)

        i = 0
        while cam.is_sequence_running() or cam.image_count > 0:
            # Check if there are images before trying to get one
            if cam.image_count > 0:
                img = cam.pop_image()
                i += 1
                cv2.imshow('', img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                cam.stop_sequence_acquisition()

            print(
                f'Remaining images: {cam.image_count:04d}',
                f'| Iterations : {i:04d}',
                end='\r',
            )

        cv2.destroyAllWindows()
    except Exception as e:
        import traceback

        traceback.print_exc()
    finally:
        PycroCore._instances[port].close()
