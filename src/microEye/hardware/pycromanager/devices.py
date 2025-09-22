from microEye.hardware.pycromanager.core import DEFAULT_BRIDGE_PORT, PycroCore
from microEye.hardware.pycromanager.enums import DeviceType, PropertyType


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


# if __name__ == '__main__':
#     import os
#     import time

#     import cv2

#     mm_path = 'C:/Program Files/Micro-Manager-2.0'
#     config = 'MMConfig_demo.cfg'
#     port = DEFAULT_BRIDGE_PORT

#     try:
#         # start_headless(
#         #     mm_path,
#         #     config,
#         #     port=port,
#         #     # python_backend=True,
#         # )

#         # stage = PycroStage(port=port)
#         # print(stage)
#         # print(stage.position)
#         # stage.move_absolute(0, 0, 50)
#         # print(stage.position)

#         cam = PycroCamera('Camera', port=port)
#         cam.exposure_current = 50

#         cam.start_sequence_acquisition(0, 500)
#         # 0 ms interval for fastest acquisition

#         # def show_image(image, metadata):
#         #     cv2.imshow('Image', image)
#         #     cv2.waitKey(1)

#         # with Acquisition(show_display=False, image_process_fn=show_image) as acq:
#         #     events = multi_d_acquisition_events(
#         #         2000,
#         #     )  # events for 2000 time points
#         #     for dic in events:
#         #         dic['camera'] = 'DCam_1'
#         #     acq.acquire(events)

#         i = 0
#         while cam.is_sequence_running() or cam.image_count > 0:
#             # Check if there are images before trying to get one
#             if cam.image_count > 0:
#                 img = cam.pop_image()
#                 i += 1
#                 cv2.imshow('', img)

#             key = cv2.waitKey(1)
#             if key == ord('q'):
#                 cam.stop_sequence_acquisition()

#             print(
#                 f'Remaining images: {cam.image_count:04d}',
#                 f'| Iterations : {i:04d}',
#                 end='\r',
#             )

#         cv2.destroyAllWindows()
#     except Exception as e:
#         import traceback

#         traceback.print_exc()
#     finally:
#         PycroCore._instances[port].close()
