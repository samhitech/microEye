import time
from typing import Any, Optional, Union

from pycromanager import Core

from microEye.hardware.pycromanager.enums import *
from microEye.hardware.pycromanager.utils import *

DEFAULT_TIMEOUT = 30  # Default timeout in seconds
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY = 5  # seconds

# def add_timeout_to_all_methods(cls):
#     for attr_name, attr_value in cls.__dict__.items():
#         if callable(attr_value) and not attr_name.startswith('__'):
#             setattr(cls, attr_name, timeout(DEFAULT_TIMEOUT)(attr_value))
#     return cls


# @add_timeout_to_all_methods
class PycroCore:
    _CORE = None
    _instance = None

    def __new__(cls, *args, **kwargs):
        # If the single instance doesn't exist, create a new one
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        # Return the single instance
        return cls._instance

    @classmethod
    def instance(cls):
        if cls._instance is None:
            return PycroCore()

        return cls._instance

    def __init__(self):
        self.connect()

    def connect(self):
        if self.is_connected():
            return

        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            try:
                self._CORE = Core(timeout=5000)
                print(f'Successfully connected on attempt {attempt + 1}')
                return
            except Exception as e:
                self._CORE = None
                print(f'Connection attempt {attempt + 1} failed: {str(e)}')
                if attempt < MAX_RECONNECT_ATTEMPTS - 1:
                    print(f'Retrying in {RECONNECT_DELAY} seconds...')
                    time.sleep(RECONNECT_DELAY)
        raise ConnectionError('Failed to connect after maximum attempts')

    def reconnect(self):
        print('Attempting to reconnect...')
        self.connect()

    def is_connected(self):
        # This is a simple check. You might want to implement a more robust check
        return PycroCore._CORE is not None and bool(self.get_version_info())

    def add_galvo_polygon_vertex(
        self, galvo_label: str, polygon_index: int, x: float, y: float
    ) -> None:
        '''TODO: Test this'''
        self._CORE.add_galvo_polygon_vertex(galvo_label, polygon_index, x, y)

    @staticmethod
    def add_search_path(path: str) -> None:
        '''TODO: Test this'''
        Core.add_search_path(path)

    # def assign_image_synchro(self, device_label: str) -> None:
    #     '''TODO: Test this'''
    #     self._CORE.assign_image_synchro(device_label)

    def clear_circular_buffer(self) -> None:
        self._CORE.clear_circular_buffer()

    def clear_roi(self) -> None:
        self._CORE.clear_roi()

    def debug_log_enabled(self) -> bool:
        return self._CORE.debug_log_enabled()

    def define_config(
        self,
        group_name: str,
        config_name: str,
        device_label: Optional[str] = None,
        prop_name: Optional[str] = None,
        value: Optional[str] = None,
    ) -> None:
        '''TODO: Test this'''
        if device_label is None and prop_name is None and value is None:
            self._CORE.define_config(group_name, config_name)
        else:
            self._CORE.define_config(
                group_name, config_name, device_label, prop_name, value
            )

    def define_config_group(self, group_name: str) -> None:
        '''TODO: Test this'''
        self._CORE.define_config_group(group_name)

    def define_pixel_size_config(
        self,
        resolution_id: str,
        device_label: Optional[str] = None,
        prop_name: Optional[str] = None,
        value: Optional[str] = None,
    ) -> None:
        '''TODO: Test this'''
        if device_label is None and prop_name is None and value is None:
            self._CORE.define_pixel_size_config(resolution_id)
        else:
            self._CORE.define_pixel_size_config(
                resolution_id, device_label, prop_name, value
            )

    def define_property_block(
        self, block_name: str, property_name: str, property_value: str
    ) -> None:
        '''TODO: Test this'''
        self._CORE.define_property_block(block_name, property_name, property_value)

    def define_state_label(
        self, state_device_label: str, state: int, state_label: str
    ) -> None:
        '''TODO: Test this'''
        self._CORE.define_state_label(state_device_label, state, state_label)

    def delete(self) -> None:
        self._CORE.delete()

    def delete_config(
        self,
        group_name: str,
        config_name: str,
        device_label: Optional[str] = None,
        prop_name: Optional[str] = None,
    ) -> None:
        '''TODO: Test this'''
        if device_label is None and prop_name is None:
            self._CORE.delete_config(group_name, config_name)
        else:
            self._CORE.delete_config(group_name, config_name, device_label, prop_name)

    def delete_config_group(self, group_name: str) -> None:
        '''TODO: Test this'''
        self._CORE.delete_config_group(group_name)

    def delete_galvo_polygons(self, galvo_label: str) -> None:
        '''TODO: Test this'''
        self._CORE.delete_galvo_polygons(galvo_label)

    def delete_pixel_size_config(self, config_name: str) -> None:
        self._CORE.delete_pixel_size_config(config_name)

    def detect_device(self, device_label: str) -> DeviceDetectionStatus:
        res = self._CORE.detect_device(device_label)
        if hasattr(res, 'swig_value'):
            return DeviceDetectionStatus(res.swig_value())

        return None

    def device_busy(self, label: str) -> bool:
        return self._CORE.device_busy(label)

    def device_type_busy(self, dev_type: Any) -> bool:
        '''TODO: dev_type should be of type mmcorej.DeviceType'''
        return self._CORE.device_type_busy(dev_type)

    def display_slm_image(self, slm_label: str) -> None:
        '''TODO: Test this'''
        self._CORE.display_slm_image(slm_label)

    def enable_continuous_focus(self, enable: bool) -> None:
        '''TODO: Test this'''
        self._CORE.enable_continuous_focus(enable)

    def enable_debug_log(self, enable: bool) -> None:
        self._CORE.enable_debug_log(enable)

    def enable_stderr_log(self, enable: bool) -> None:
        self._CORE.enable_stderr_log(enable)

    def full_focus(self) -> None:
        self._CORE.full_focus()

    def get_allowed_property_values(self, label: str, prop_name: str) -> list[str]:
        '''
        Get the allowed property values for a given device.

        Parameters:
        ----------
        label : str
            The label of the device.
        prop_name : str
            The name of the property.

        Returns:
        -------
        list[str]
            A list of allowed property values.

        '''
        return vec_to_list(self._CORE.get_allowed_property_values(label, prop_name))

    def get_api_version_info(self) -> str:
        return self._CORE.get_api_version_info()

    def get_auto_focus_device(self) -> str:
        return self._CORE.get_auto_focus_device()

    def get_auto_focus_offset(self) -> float:
        '''TODO: remove or fix freezing'''
        return self._CORE.get_auto_focus_offset()

    def get_auto_shutter(self) -> bool:
        return self._CORE.get_auto_shutter()

    def get_available_config_groups(self) -> list[str]:
        return vec_to_list(self._CORE.get_available_config_groups())

    def get_available_configs(self, config_group: str) -> list[str]:
        return vec_to_list(self._CORE.get_available_configs(config_group))

    def get_available_device_descriptions(self, library: str) -> list[str]:
        '''TODO: Test this'''
        '''TODO: remove or fix freezing'''
        return vec_to_list(self._CORE.get_available_device_descriptions(library))

    def get_available_devices(self, library: str) -> list[str]:
        '''TODO: Test this'''
        return vec_to_list(self._CORE.get_available_devices(library))

    def get_available_device_types(self, library: str) -> list[int]:
        '''TODO: Test this'''
        return self._CORE.get_available_device_types(library)

    def get_available_pixel_size_configs(self) -> list[str]:
        return vec_to_list(self._CORE.get_available_pixel_size_configs())

    # def get_available_property_blocks(self) -> list[str]:
    #     return vec_to_list(self._core.get_available_property_blocks())

    def get_buffer_free_capacity(self) -> int:
        return self._CORE.get_buffer_free_capacity()

    def get_buffer_total_capacity(self) -> int:
        return self._CORE.get_buffer_total_capacity()

    def get_bytes_per_pixel(self) -> int:
        return self._CORE.get_bytes_per_pixel()

    def get_camera_channel_name(self, channel_nr: int) -> str:
        return self._CORE.get_camera_channel_name(channel_nr)

    def get_camera_device(self) -> str:
        return self._CORE.get_camera_device()

    def get_channel_group(self) -> str:
        return self._CORE.get_channel_group()

    def get_circular_buffer_memory_footprint(self) -> int:
        return self._CORE.get_circular_buffer_memory_footprint()

    def get_config_data(self, config_group: str, config_name: str) -> Any:
        '''TODO: remove or fix freezing'''
        return self._CORE.get_config_data(config_group, config_name)

    def get_config_group_state(self, group: str) -> Any:
        return self._CORE.get_config_group_state(group)

    def get_config_group_state_from_cache(self, group: str) -> Any:
        return self._CORE.get_config_group_state_from_cache(group)

    def get_config_state(self, group: str, config: str) -> Any:
        '''TODO: remove or fix freezing'''
        return self._CORE.get_config_state(group, config)

    def get_core_error_text(self, code: int) -> str:
        return self._CORE.get_core_error_text(code)

    def get_current_config(self, group_name: str) -> str:
        return self._CORE.get_current_config(group_name)

    def get_current_config_from_cache(self, group_name: str) -> str:
        return self._CORE.get_current_config_from_cache(group_name)

    def get_current_focus_score(self) -> float:
        return self._CORE.get_current_focus_score()

    def get_current_pixel_size_config(self, cached: bool = False) -> str:
        return self._CORE.get_current_pixel_size_config(cached)

    def get_data(self, state_device_label: str) -> Any:
        return self._CORE.get_data(state_device_label)

    def get_device_adapter_names(self) -> list[str]:
        return vec_to_list(self._CORE.get_device_adapter_names())

    def get_device_adapter_search_paths(self) -> list[str]:
        return vec_to_list(self._CORE.get_device_adapter_search_paths())

    def get_device_delay_ms(self, label: str) -> float:
        return self._CORE.get_device_delay_ms(label)

    def get_device_description(self, label: str) -> str:
        return self._CORE.get_device_description(label)

    # @staticmethod
    # def get_device_libraries() -> list[str]:
    #     return vec_to_list(Core.get_device_libraries())

    # def get_device_library(self, label: str) -> str:
    #     return self._core.get_device_library(label)

    def get_device_name(self, label: str) -> str:
        return self._CORE.get_device_name(label)

    def get_device_property_names(self, label: str) -> list[str]:
        return vec_to_list(self._CORE.get_device_property_names(label))

    def get_device_type(self, label: str) -> DeviceType:
        res = self._CORE.get_device_type(label)
        if res and hasattr(res, 'swig_value'):
            return DeviceType(res.swig_value())

        return None

    def get_exposure(self, label: Optional[str] = None) -> float:
        if label is None:
            return self._CORE.get_exposure()
        else:
            return self._CORE.get_exposure(label)

    def get_exposure_sequence_max_length(self, camera_label: str) -> int:
        '''TODO: remove or fix freezing'''
        return self._CORE.get_exposure_sequence_max_length(camera_label)

    def get_focus_device(self) -> str:
        return self._CORE.get_focus_device()

    def get_focus_direction(self, stage_label: str) -> FocusDirection:
        return FocusDirection(self._CORE.get_focus_direction(stage_label))

    def get_galvo_channel(self, galvo_label: str) -> str:
        '''TODO: Test this'''
        return self._CORE.get_galvo_channel(galvo_label)

    def get_galvo_device(self) -> str:
        return self._CORE.get_galvo_device()

    def get_galvo_position(self, galvo_device: str) -> tuple[float, float]:
        '''TODO: Test this'''
        return self._CORE.get_galvo_position(galvo_device)

    def get_galvo_x_minimum(self, galvo_label: str) -> float:
        '''TODO: Test this'''
        return self._CORE.get_galvo_x_minimum(galvo_label)

    def get_galvo_x_range(self, galvo_label: str) -> float:
        '''TODO: Test this'''
        return self._CORE.get_galvo_x_range(galvo_label)

    def get_galvo_y_minimum(self, galvo_label: str) -> float:
        '''TODO: Test this'''
        return self._CORE.get_galvo_y_minimum(galvo_label)

    def get_galvo_y_range(self, galvo_label: str) -> float:
        '''TODO: Test this'''
        return self._CORE.get_galvo_y_range(galvo_label)

    # def get_host_name(self) -> str:
    #     return self._core.get_host_name()

    def get_image(self, num_channel: Optional[int] = None) -> Any:
        if num_channel is None:
            return self._CORE.get_image()
        else:
            return self._CORE.get_image(num_channel)

    def get_image_bit_depth(self) -> int:
        return self._CORE.get_image_bit_depth()

    def get_image_buffer_size(self) -> int:
        return self._CORE.get_image_buffer_size()

    def get_image_height(self) -> int:
        return self._CORE.get_image_height()

    def get_image_processor_device(self) -> str:
        return self._CORE.get_image_processor_device()

    def get_image_width(self) -> int:
        return self._CORE.get_image_width()

    def get_installed_device_description(
        self, hub_label: str, peripheral_label: str
    ) -> str:
        return self._CORE.get_installed_device_description(hub_label, peripheral_label)

    def get_installed_devices(self, hub_label: str) -> list[str]:
        return vec_to_list(self._CORE.get_installed_devices(hub_label))

    def get_last_focus_score(self) -> float:
        return self._CORE.get_last_focus_score()

    def get_last_image(self) -> Any:
        return self._CORE.get_last_image()

    def get_last_image_md(
        self,
        channel: Optional[int] = None,
        slice: Optional[int] = None,
        md: Optional[Any] = None,
    ) -> Any:
        '''TODO: Test this'''
        if channel is None and slice is None:
            return self._CORE.get_last_image_md(md)
        else:
            return self._CORE.get_last_image_md(channel, slice, md)

    def get_last_tagged_image(self, camera_channel_index: Optional[int] = None) -> Any:
        '''TODO: Test this'''
        if camera_channel_index is None:
            return self._CORE.get_last_tagged_image()
        else:
            return self._CORE.get_last_tagged_image(camera_channel_index)

    def get_loaded_devices(self) -> list[str]:
        return vec_to_list(self._CORE.get_loaded_devices())

    def get_loaded_devices_of_type(self, dev_type: DeviceType) -> list[str]:
        type = self._CORE._bridge._get_java_class('mmcorej.DeviceType').swig_to_enum(
            dev_type.value
        )
        return vec_to_list(self._CORE.get_loaded_devices_of_type(type))

    def get_loaded_peripheral_devices(self, hub_label: str) -> list[str]:
        return vec_to_list(self._CORE.get_loaded_peripheral_devices(hub_label))

    # def get_mac_addresses(self) -> list[str]:
    #     return vec_to_list(self._core.get_mac_addresses())

    def get_magnification_factor(self) -> float:
        return self._CORE.get_magnification_factor()

    def get_multi_roi(self) -> list[Any]:
        return vec_to_list(self._CORE.get_multi_roi())

    def get_n_before_last_image_md(self, n: int, md: Any) -> Any:
        '''TODO: Test this'''
        return self._CORE.get_n_before_last_image_md(n, md)

    def get_n_before_last_tagged_image(self, n: int) -> Any:
        '''TODO: Test this'''
        return self._CORE.get_n_before_last_tagged_image(n)

    def get_number_of_camera_channels(self) -> int:
        return self._CORE.get_number_of_camera_channels()

    def get_number_of_components(self) -> int:
        return self._CORE.get_number_of_components()

    # def get_number_of_devices(self) -> int:
    #     return self._core.get_number_of_devices()

    # def get_number_of_device_types(self) -> int:
    #     return self._core.get_number_of_device_types()

    def get_number_of_galvo_polygons(self, galvo_label: str) -> int:
        '''TODO: Test this'''
        return self._CORE.get_number_of_galvo_polygons(galvo_label)

    # def get_number_of_images(self) -> int:
    #     return self._core.get_number_of_images()

    def get_number_of_states(self, state_device_label: str) -> int:
        '''TODO: Test this'''
        return self._CORE.get_number_of_states(state_device_label)

    def get_parent_label(self, peripheral_label: str) -> str:
        '''TODO: Test this'''
        return self._CORE.get_parent_label(peripheral_label)

    def get_pixel_size_affine(self, cached: bool = False) -> list[float]:
        return vec_to_list(self._CORE.get_pixel_size_affine(cached))

    def get_pixel_size_affine_as_string(self) -> str:
        return self._CORE.get_pixel_size_affine_as_string()

    def get_pixel_size_affine_by_id(self, resolution_id: str) -> list[float]:
        '''TODO: freezes need to find resolution id'''
        return vec_to_list(self._CORE.get_pixel_size_affine_by_id(resolution_id))

    def get_pixel_size_config_data(self, config_name: str) -> Any:
        '''TODO: freezes need to find config name = resolution id, e.g., Res10x'''
        return config_to_dict(self._CORE.get_pixel_size_config_data(config_name))

    def get_pixel_size_um(self, cached: bool = False) -> float:
        return self._CORE.get_pixel_size_um(cached)

    def get_pixel_size_um_by_id(self, resolution_id: str) -> float:
        return self._CORE.get_pixel_size_um_by_id(resolution_id)

    def get_position(self, stage_label: Optional[str] = None) -> float:
        if stage_label is None:
            return self._CORE.get_position()
        else:
            return self._CORE.get_position(stage_label)

    def get_primary_log_file(self) -> str:
        return self._CORE.get_primary_log_file()

    def get_property(self, label: str, prop_name: str) -> str:
        return self._CORE.get_property(label, prop_name)

    # def get_property_block_data(self, block_name: str) -> Any:
    #     return self._core.get_property_block_data(block_name)

    def get_property_from_cache(self, device_label: str, prop_name: str) -> str:
        return self._CORE.get_property_from_cache(device_label, prop_name)

    def get_property_lower_limit(self, label: str, prop_name: str) -> float:
        return self._CORE.get_property_lower_limit(label, prop_name)

    def get_property_sequence_max_length(self, label: str, prop_name: str) -> int:
        '''TODO: freezes'''
        return self._CORE.get_property_sequence_max_length(label, prop_name)

    def get_property_type(self, label: str, prop_name: str) -> PropertyType:
        return PropertyType.from_java(self._CORE.get_property_type(label, prop_name))

    def get_property_upper_limit(self, label: str, prop_name: str) -> float:
        return self._CORE.get_property_upper_limit(label, prop_name)

    def get_remaining_image_count(self) -> int:
        return self._CORE.get_remaining_image_count()

    def get_roi(self, label: Optional[str] = None) -> tuple[int, int, int, int]:
        '''
        Get the region of interest (ROI) coordinates.

        Parameters
        ----------
        label : str, optional
            The label of the ROI. If None, the default ROI will be used.

        Returns
        -------
        tuple
            A tuple containing the x, y, width, and height of the ROI.
        '''
        roi = self._CORE.get_roi() if label is None else self._CORE.get_roi(label)

        return roi.x, roi.y, roi.width, roi.height

    def get_serial_port_answer(self, port_label: str, term: str) -> str:
        '''TODO: to be tested'''
        return self._CORE.get_serial_port_answer(port_label, term)

    def get_shutter_device(self) -> str:
        return self._CORE.get_shutter_device()

    def get_shutter_open(self, shutter_label: Optional[str] = None) -> bool:
        if shutter_label is None:
            return self._CORE.get_shutter_open()
        else:
            return self._CORE.get_shutter_open(shutter_label)

    def get_slm_bytes_per_pixel(self, slm_label: str) -> int:
        '''TODO: to be tested'''
        return self._CORE.get_slm_bytes_per_pixel(slm_label)

    def get_slm_device(self) -> str:
        return self._CORE.get_slm_device()

    def get_slm_exposure(self, slm_label: str) -> float:
        '''TODO: to be tested'''
        return self._CORE.get_slm_exposure(slm_label)

    def get_slm_height(self, slm_label: str) -> int:
        '''TODO: to be tested'''
        return self._CORE.get_slm_height(slm_label)

    def get_slm_number_of_components(self, slm_label: str) -> int:
        '''TODO: to be tested'''
        return self._CORE.get_slm_number_of_components(slm_label)

    def get_slm_sequence_max_length(self, slm_label: str) -> int:
        '''TODO: to be tested'''
        return self._CORE.get_slm_sequence_max_length(slm_label)

    def get_slm_width(self, slm_label: str) -> int:
        '''TODO: to be tested'''
        return self._CORE.get_slm_width(slm_label)

    def get_stage_sequence_max_length(self, stage_label: str) -> int:
        '''TODO: freezes'''
        return self._CORE.get_stage_sequence_max_length(stage_label)

    def get_state(self, state_device_label: str) -> int:
        return self._CORE.get_state(state_device_label)

    def get_state_from_label(self, state_device_label: str, state_label: str) -> int:
        return self._CORE.get_state_from_label(state_device_label, state_label)

    def get_state_label(self, state_device_label: str) -> str:
        return self._CORE.get_state_label(state_device_label)

    # def get_state_label_data(self, state_device_label: str, state_label: str) -> Any:
    #     return self._core.get_state_label_data(state_device_label, state_label)

    def get_state_labels(self, state_device_label: str) -> list[str]:
        return vec_to_list(self._CORE.get_state_labels(state_device_label))

    def get_system_state(self) -> dict:
        return group_config_dict(config_to_dict(self._CORE.get_system_state()))

    def get_system_state_cache(self) -> Any:
        return group_config_dict(config_to_dict(self._CORE.get_system_state_cache()))

    def get_tagged_image(self, camera_channel_index: Optional[int] = None) -> Any:
        if camera_channel_index is None:
            return self._CORE.get_tagged_image()
        else:
            return self._CORE.get_tagged_image(camera_channel_index)

    def get_timeout_ms(self) -> int:
        return self._CORE.get_timeout_ms()

    # def get_user_id(self) -> str:
    #     return self._core.get_user_id()

    def get_version_info(self) -> str:
        return self._CORE.get_version_info()

    def get_x_position(self, xy_stage_label: Optional[str] = None) -> float:
        if xy_stage_label is None:
            return self._CORE.get_x_position()
        else:
            return self._CORE.get_x_position(xy_stage_label)

    def get_xy_position(
        self, x_stage, y_stage, xy_stage_label: Optional[str] = None
    ) -> tuple[float, float]:
        '''TODO: test and understand'''
        if xy_stage_label is None:
            return self._CORE.get_xy_position(x_stage, y_stage)
        else:
            return self._CORE.get_xy_position(xy_stage_label, x_stage, y_stage)

    def get_xy_stage_device(self) -> str:
        return self._CORE.get_xy_stage_device()

    def get_xy_stage_position(self, stage: Optional[str] = None) -> tuple[float, float]:
        '''
        Get the XY stage position.

        Parameters
        ----------
        stage : str, optional
            The name of the stage. If None, the default stage will be used.

        Returns
        -------
        xy_position : tuple[float, float]
            The X and Y coordinates of the stage position.
        '''
        if stage is None:
            res = self._CORE.get_xy_stage_position()
        else:
            res = self._CORE.get_xy_stage_position(stage)

        return res.x, res.y

    def get_xy_stage_sequence_max_length(self, xy_stage_label: str) -> int:
        '''TODO: freezes, test and understand'''
        return self._CORE.get_xy_stage_sequence_max_length(xy_stage_label)

    def get_y_position(self, xy_stage_label: Optional[str] = None) -> float:
        if xy_stage_label is None:
            return self._CORE.get_y_position()
        else:
            return self._CORE.get_y_position(xy_stage_label)

    def has_property(self, label: str, prop_name: str) -> bool:
        return self._CORE.has_property(label, prop_name)

    def has_property_limits(self, label: str, prop_name: str) -> bool:
        return self._CORE.has_property_limits(label, prop_name)

    def home(self, xy_or_z_stage_label: str) -> None:
        '''TODO: freezes, test and understand'''
        self._CORE.home(xy_or_z_stage_label)

    def incremental_focus(self) -> None:
        '''TODO: freezes, test and understand'''
        self._CORE.incremental_focus()

    def initialize_all_devices(self) -> None:
        '''TODO: freezes, test and understand'''
        self._CORE.initialize_all_devices()

    def initialize_circular_buffer(self) -> None:
        self._CORE.initialize_circular_buffer()

    def initialize_device(self, label: str) -> None:
        '''TODO: freezes, test and understand'''
        self._CORE.initialize_device(label)

    def is_buffer_overflowed(self) -> bool:
        return self._CORE.is_buffer_overflowed()

    def is_config_defined(self, group_name: str, config_name: str) -> bool:
        return self._CORE.is_config_defined(group_name, config_name)

    def is_continuous_focus_drive(self, stage_label: str) -> bool:
        return self._CORE.is_continuous_focus_drive(stage_label)

    def is_continuous_focus_enabled(self) -> bool:
        return self._CORE.is_continuous_focus_enabled()

    def is_continuous_focus_locked(self) -> bool:
        return self._CORE.is_continuous_focus_locked()

    def is_exposure_sequenceable(self, camera_label: str) -> bool:
        return self._CORE.is_exposure_sequenceable(camera_label)

    def is_group_defined(self, group_name: str) -> bool:
        return self._CORE.is_group_defined(group_name)

    def is_multi_roi_enabled(self) -> bool:
        return self._CORE.is_multi_roi_enabled()

    def is_multi_roi_supported(self) -> bool:
        return self._CORE.is_multi_roi_supported()

    def is_pixel_size_config_defined(self, resolution_id: str) -> bool:
        return self._CORE.is_pixel_size_config_defined(resolution_id)

    def is_property_pre_init(self, label: str, prop_name: str) -> bool:
        return self._CORE.is_property_pre_init(label, prop_name)

    def is_property_read_only(self, label: str, prop_name: str) -> bool:
        return self._CORE.is_property_read_only(label, prop_name)

    def is_property_sequenceable(self, label: str, prop_name: str) -> bool:
        return self._CORE.is_property_sequenceable(label, prop_name)

    def is_sequence_running(self, camera_label: Optional[str] = None) -> bool:
        if camera_label is None:
            return self._CORE.is_sequence_running()
        else:
            return self._CORE.is_sequence_running(camera_label)

    def is_stage_linear_sequenceable(self, stage_label: str) -> bool:
        return self._CORE.is_stage_linear_sequenceable(stage_label)

    def is_stage_sequenceable(self, stage_label: str) -> bool:
        return self._CORE.is_stage_sequenceable(stage_label)

    def is_xy_stage_sequenceable(self, xy_stage_label: str) -> bool:
        return self._CORE.is_xy_stage_sequenceable(xy_stage_label)

    def load_device(self, label: str, module_name: str, device_name: str) -> None:
        '''TODO: freezes, test and understand'''
        self._CORE.load_device(label, module_name, device_name)

    def load_exposure_sequence(
        self, camera_label: str, exposure_sequence_ms: list[float]
    ) -> None:
        if self.is_exposure_sequenceable():
            self._CORE.load_exposure_sequence(camera_label, exposure_sequence_ms)
        else:
            raise ValueError(
                f'Exposure sequence is not supported for camera {camera_label}'
            )

    def load_galvo_polygons(self, galvo_label: str) -> None:
        '''TODO: freezes, test and understand'''
        self._CORE.load_galvo_polygons(galvo_label)

    def load_property_sequence(
        self, label: str, prop_name: str, event_sequence: list[str]
    ) -> None:
        if self.is_property_sequenceable(label, prop_name):
            self._CORE.load_property_sequence(label, prop_name, event_sequence)
        else:
            raise ValueError(
                f'Property sequence for {label} property {prop_name} is not supported'
            )

    def load_slm_sequence(self, slm_label: str, image_sequence: list[bytes]) -> None:
        '''TODO: freezes, test and understand'''
        self._CORE.load_slm_sequence(slm_label, image_sequence)

    def load_stage_sequence(
        self, stage_label: str, position_sequence: list[float]
    ) -> None:
        if self.is_stage_sequenceable(stage_label):
            self._CORE.load_stage_sequence(stage_label, position_sequence)
        else:
            raise ValueError(f'Stage sequence for {stage_label} is not supported')

    def load_system_configuration(self, file_name: str) -> None:
        '''TODO: test'''
        self._CORE.load_system_configuration(file_name)

    def load_system_state(self, file_name: str) -> None:
        '''TODO: test'''
        self._CORE.load_system_state(file_name)

    def load_xy_stage_sequence(
        self, xy_stage_label: str, x_sequence: list[float], y_sequence: list[float]
    ) -> None:
        if self.is_xy_stage_sequenceable(xy_stage_label):
            self._CORE.load_xy_stage_sequence(xy_stage_label, x_sequence, y_sequence)
        else:
            raise ValueError(f'XY stage sequence for {xy_stage_label} is not supported')

    def log_message(self, msg: str, debug_only: bool = False) -> None:
        self._CORE.log_message(msg, debug_only)

    def point_galvo_and_fire(
        self, galvo_label: str, x: float, y: float, pulse_time_us: float
    ) -> None:
        '''TODO: test'''
        self._CORE.point_galvo_and_fire(galvo_label, x, y, pulse_time_us)

    def pop_next_image(self) -> Any:
        '''TODO: test'''
        return self._CORE.pop_next_image()

    def pop_next_image_md(
        self,
        channel: Optional[int] = None,
        slice: Optional[int] = None,
        md: Optional[Any] = None,
    ) -> Any:
        '''TODO: test'''
        if channel is None and slice is None:
            return self._CORE.pop_next_image_md(md)
        else:
            return self._CORE.pop_next_image_md(channel, slice, md)

    def pop_next_tagged_image(self, camera_channel_index: Optional[int] = None) -> Any:
        '''TODO: test'''
        if camera_channel_index is None:
            return self._CORE.pop_next_tagged_image()
        else:
            return self._CORE.pop_next_tagged_image(camera_channel_index)

    def prepare_sequence_acquisition(self, camera_label: str) -> None:
        self._CORE.prepare_sequence_acquisition(camera_label)

    def read_from_serial_port(self, port_label: str) -> bytes:
        '''TODO: test, freezes'''
        return self._CORE.read_from_serial_port(port_label)

    def register_callback(self, cb: Any) -> None:
        '''TODO: test'''
        self._CORE.register_callback(cb)

    # def remove_image_synchro(self, device_label: str) -> None:
    #     '''TODO: test, freezes'''
    #     self._CORE.remove_image_synchro(device_label)

    # def remove_image_synchro_all(self) -> None:
    #     self._core.remove_image_synchro_all()

    def rename_config(
        self, group_name: str, old_config_name: str, new_config_name: str
    ) -> None:
        '''TODO: test'''
        self._CORE.rename_config(group_name, old_config_name, new_config_name)

    def rename_config_group(self, old_group_name: str, new_group_name: str) -> None:
        '''TODO: test'''
        self._CORE.rename_config_group(old_group_name, new_group_name)

    def rename_pixel_size_config(
        self, old_config_name: str, new_config_name: str
    ) -> None:
        '''TODO: test'''
        self._CORE.rename_pixel_size_config(old_config_name, new_config_name)

    def reset(self) -> None:
        self._CORE.reset()

    def run_galvo_polygons(self, galvo_label: str) -> None:
        '''TODO: test'''
        self._CORE.run_galvo_polygons(galvo_label)

    def run_galvo_sequence(self, galvo_label: str) -> None:
        '''TODO: test'''
        self._CORE.run_galvo_sequence(galvo_label)

    def save_system_configuration(self, file_name: str) -> None:
        '''TODO: freezes'''
        self._CORE.save_system_configuration(file_name)

    def save_system_state(self, file_name: str) -> None:
        '''TODO: freezes'''
        self._CORE.save_system_state(file_name)

    def set_adapter_origin(
        self, new_z_um: float, stage_label: Optional[str] = None
    ) -> None:
        '''TODO: freezes'''
        if stage_label is None:
            self._CORE.set_adapter_origin(new_z_um)
        else:
            self._CORE.set_adapter_origin(stage_label, new_z_um)

    def set_adapter_origin_xy(
        self, new_x_um: float, new_y_um: float, xy_stage_label: Optional[str] = None
    ) -> None:
        '''TODO: freezes'''
        if xy_stage_label is None:
            self._CORE.set_adapter_origin_xy(new_x_um, new_y_um)
        else:
            self._CORE.set_adapter_origin_xy(xy_stage_label, new_x_um, new_y_um)

    def set_auto_focus_device(self, focus_label: str) -> None:
        '''TODO: test'''
        self._CORE.set_auto_focus_device(focus_label)

    def set_auto_focus_offset(self, offset: float) -> None:
        self._CORE.set_auto_focus_offset(offset)

    def set_auto_shutter(self, state: bool) -> None:
        self._CORE.set_auto_shutter(state)

    def set_camera_device(self, camera_label: str) -> None:
        self._CORE.set_camera_device(camera_label)

    def set_channel_group(self, channel_group: str) -> None:
        self._CORE.set_channel_group(channel_group)

    def set_circular_buffer_memory_footprint(self, size_mb: int) -> None:
        self._CORE.set_circular_buffer_memory_footprint(size_mb)

    def set_config(self, group_name: str, config_name: str) -> None:
        self._CORE.set_config(group_name, config_name)

    def set_device_adapter_search_paths(self, paths: list[str]) -> None:
        '''TODO: test'''
        self._CORE.set_device_adapter_search_paths(paths)

    def set_device_delay_ms(self, label: str, delay_ms: float) -> None:
        self._CORE.set_device_delay_ms(label, delay_ms)

    def set_exposure(self, exp: float, camera_label: Optional[str] = None) -> None:
        if camera_label is None:
            self._CORE.set_exposure(exp)
        else:
            self._CORE.set_exposure(camera_label, exp)

    def set_focus_device(self, focus_label: str) -> None:
        self._CORE.set_focus_device(focus_label)

    def set_focus_direction(self, stage_label: str, sign: int) -> None:
        self._CORE.set_focus_direction(stage_label, sign)

    def set_galvo_device(self, galvo_label: str) -> None:
        '''TODO: test'''
        self._CORE.set_galvo_device(galvo_label)

    def set_galvo_illumination_state(self, galvo_label: str, on: bool) -> None:
        '''TODO: test'''
        self._CORE.set_galvo_illumination_state(galvo_label, on)

    def set_galvo_polygon_repetitions(self, galvo_label: str, repetitions: int) -> None:
        '''TODO: test'''
        self._CORE.set_galvo_polygon_repetitions(galvo_label, repetitions)

    def set_galvo_position(self, galvo_label: str, x: float, y: float) -> None:
        '''TODO: test'''
        self._CORE.set_galvo_position(galvo_label, x, y)

    def set_galvo_spot_interval(self, galvo_label: str, pulse_time_us: float) -> None:
        '''TODO: test'''
        self._CORE.set_galvo_spot_interval(galvo_label, pulse_time_us)

    def set_image_processor_device(self, proc_label: str) -> None:
        '''TODO: test'''
        self._CORE.set_image_processor_device(proc_label)

    def set_multi_roi(self, rois: list[Any]) -> None:
        '''TODO: test'''
        self._CORE.set_multi_roi(rois)

    def set_origin(self, stage_label: Optional[str] = None) -> None:
        if stage_label is None:
            self._CORE.set_origin()
        else:
            self._CORE.set_origin(stage_label)

    def set_origin_x(self, xy_stage_label: Optional[str] = None) -> None:
        if xy_stage_label is None:
            self._CORE.set_origin_x()
        else:
            self._CORE.set_origin_x(xy_stage_label)

    def set_origin_xy(self, xy_stage_label: Optional[str] = None) -> None:
        if xy_stage_label is None:
            self._CORE.set_origin_xy()
        else:
            self._CORE.set_origin_xy(xy_stage_label)

    def set_origin_y(self, xy_stage_label: Optional[str] = None) -> None:
        if xy_stage_label is None:
            self._CORE.set_origin_y()
        else:
            self._CORE.set_origin_y(xy_stage_label)

    def set_parent_label(self, device_label: str, parent_hub_label: str) -> None:
        '''TODO: test'''
        self._CORE.set_parent_label(device_label, parent_hub_label)

    def set_pixel_size_affine(self, resolution_id: str, affine: list[float]) -> None:
        '''TODO: Error does not accept list, needs mmcorej.DoubleVector'''
        self._CORE.set_pixel_size_affine(resolution_id, affine)

    def set_pixel_size_config(self, resolution_id: str) -> None:
        self._CORE.set_pixel_size_config(resolution_id)

    def set_pixel_size_um(self, resolution_id: str, pix_size: float) -> None:
        self._CORE.set_pixel_size_um(resolution_id, pix_size)

    def set_position(self, position: float, stage_label: Optional[str] = None) -> None:
        if stage_label is None:
            self._CORE.set_position(position)
        else:
            self._CORE.set_position(stage_label, position)

    def set_primary_log_file(self, filename: str, truncate: bool = False) -> None:
        '''TODO: test'''
        self._CORE.set_primary_log_file(filename, truncate)

    def set_property(self, label: str, prop_name: str, prop_value: Any) -> None:
        if self.is_property_read_only(label, prop_name):
            raise ValueError(f'Property {prop_name} is read-only for device {label}')

        if not self.has_property(label, prop_name):
            raise ValueError(f'Property {prop_name} does not exist for device {label}')

        type = self.get_property_type(label, prop_name).to_python()
        if isinstance(prop_value, type):
            if self.has_property_limits(label, prop_name):
                prop_min = self.get_property_lower_limit(label, prop_name)
                prop_max = self.get_property_upper_limit(label, prop_name)
                prop_value = max(prop_min, min(prop_value, prop_max))

            allowed = self.get_allowed_property_values(label, prop_name)
            if allowed and str(prop_value) not in allowed:
                raise ValueError(
                    f'Property {prop_name} must be one of {allowed}, not {prop_value}'
                )

            self._CORE.set_property(label, prop_name, str(prop_value))
        else:
            raise ValueError(f'Property {prop_name} must be of type {type}')

    def set_relative_position(
        self, d: float, stage_label: Optional[str] = None
    ) -> None:
        if stage_label is None:
            self._CORE.set_relative_position(d)
        else:
            self._CORE.set_relative_position(stage_label, d)

    def set_relative_xy_position(
        self, dx: float, dy: float, xy_stage_label: Optional[str] = None
    ) -> None:
        if xy_stage_label is None:
            self._CORE.set_relative_xy_position(dx, dy)
        else:
            self._CORE.set_relative_xy_position(xy_stage_label, dx, dy)

    def set_roi(
        self, x: int, y: int, x_size: int, y_size: int, label: Optional[str] = None
    ) -> None:
        if label is None:
            self._CORE.set_roi(x, y, x_size, y_size)
        else:
            self._CORE.set_roi(label, x, y, x_size, y_size)

    # def set_serial_port_command(
    #         self, port_label: str, command: str, term: str) -> None:
    #     self._CORE.set_serial_port_command(port_label, command, term)

    # def set_serial_properties(
    #     self,
    #     port_name: str,
    #     answer_timeout: str,
    #     baud_rate: str,
    #     delay_between_chars_ms: str,
    #     handshaking: str,
    #     parity: str,
    #     stop_bits: str,
    # ) -> None:
    #     self._CORE.set_serial_properties(
    #         port_name,
    #         answer_timeout,
    #         baud_rate,
    #         delay_between_chars_ms,
    #         handshaking,
    #         parity,
    #         stop_bits,
    #     )

    def set_shutter_device(self, shutter_label: str) -> None:
        self._CORE.set_shutter_device(shutter_label)

    def set_shutter_open(
        self, state: bool, shutter_label: Optional[str] = None
    ) -> None:
        if shutter_label is None:
            self._CORE.set_shutter_open(state)
        else:
            self._CORE.set_shutter_open(shutter_label, state)

    def set_slm_device(self, slm_label: str) -> None:
        '''TODO: test'''
        self._CORE.set_slm_device(slm_label)

    def set_slm_exposure(self, slm_label: str, exposure_ms: float) -> None:
        '''TODO: test'''
        self._CORE.set_slm_exposure(slm_label, exposure_ms)

    def set_slm_image(self, slm_label: str, pixels: Union[bytes, list[int]]) -> None:
        '''TODO: test'''
        self._CORE.set_slm_image(slm_label, pixels)

    def set_slm_pixels_to(
        self,
        slm_label: str,
        intensity: int,
        red: Optional[int] = None,
        green: Optional[int] = None,
        blue: Optional[int] = None,
    ) -> None:
        '''TODO: test'''
        if red is None and green is None and blue is None:
            self._CORE.set_slm_pixels_to(slm_label, intensity)
        else:
            self._CORE.set_slm_pixels_to(slm_label, red, green, blue)

    def set_stage_linear_sequence(
        self, stage_label: str, dz_um: float, n_slices: int
    ) -> None:
        if not self.is_stage_linear_sequenceable(stage_label):
            raise ValueError(f'Stage "{stage_label}" does not support linear sequences')
        self._CORE.set_stage_linear_sequence(stage_label, dz_um, n_slices)

    def set_state(self, state_device_label: str, state: int) -> None:
        if self.get_device_type(state_device_label) != DeviceType.StateDevice:
            raise ValueError(f'Device "{state_device_label}" is not a state device')

        n_states = self.get_number_of_states(state_device_label)
        if state < 0 or state >= n_states:
            raise ValueError(f'State must be between 0 and {n_states - 1}, not {state}')
        self._CORE.set_state(state_device_label, state)

    def set_state_label(self, state_device_label: str, state_label: str) -> None:
        if self.get_device_type(state_device_label) != DeviceType.StateDevice:
            raise ValueError(f'Device "{state_device_label}" is not a state device')
        allowed = self.get_allowed_property_values(state_device_label, 'Label')
        if allowed and state_label not in allowed:
            raise ValueError(f'State label must be one of {allowed}, not {state_label}')
        self._CORE.set_state_label(state_device_label, state_label)

    def set_system_state(self, conf: Any) -> None:
        '''TODO: most likely conf is expecting an mmcore java object'''
        self._CORE.set_system_state(conf)

    def set_timeout_ms(self, timeout_ms: int) -> None:
        self._CORE.set_timeout_ms(timeout_ms)

    def set_xy_position(
        self, x: float, y: float, xy_stage_label: Optional[str] = None
    ) -> None:
        if xy_stage_label is None:
            self._CORE.set_xy_position(x, y)
        else:
            self._CORE.set_xy_position(xy_stage_label, x, y)

    def set_xy_stage_device(self, xy_stage_label: str) -> None:
        self._CORE.set_xy_stage_device(xy_stage_label)

    def sleep(self, interval_ms: float) -> None:
        self._CORE.sleep(interval_ms)

    def snap_image(self) -> None:
        self._CORE.snap_image()

    def start_continuous_sequence_acquisition(self, interval_ms: float) -> None:
        '''TODO: test, most likely requires setting up the acq'''
        self._CORE.start_continuous_sequence_acquisition(interval_ms)

    def start_exposure_sequence(self, camera_label: str) -> None:
        if not self.is_exposure_sequenceable(camera_label):
            raise ValueError(
                f'Exposure sequence is not supported for camera {camera_label}'
            )

        self._CORE.start_exposure_sequence(camera_label)

    def start_property_sequence(self, label: str, prop_name: str) -> None:
        if not self.is_property_sequenceable(label, prop_name):
            raise ValueError(
                f'Property sequence for {label} property {prop_name} is not supported'
            )

        self._CORE.start_property_sequence(label, prop_name)

    def start_secondary_log_file(
        self,
        filename: str,
        enable_debug: bool,
        truncate: bool = False,
        synchronous: bool = False,
    ) -> int:
        return self._CORE.start_secondary_log_file(
            filename, enable_debug, truncate, synchronous
        )

    def start_sequence_acquisition(
        self,
        num_images: int,
        interval_ms: float,
        stop_on_overflow: bool,
        camera_label: Optional[str] = None,
    ) -> None:
        if camera_label is None:
            self._CORE.start_sequence_acquisition(
                num_images, interval_ms, stop_on_overflow
            )
        else:
            self._CORE.start_sequence_acquisition(
                camera_label, num_images, interval_ms, stop_on_overflow
            )

    def start_slm_sequence(self, slm_label: str) -> None:
        '''TODO: test'''
        if not self.is_slm_sequenceable(slm_label):
            raise ValueError(f'SLM sequence for {slm_label} is not supported')

        self._CORE.start_slm_sequence(slm_label)

    def start_stage_sequence(self, stage_label: str) -> None:
        '''TODO: test'''
        if not self.is_stage_sequenceable(stage_label):
            raise ValueError(f'Stage sequence for {stage_label} is not supported')

        self._CORE.start_stage_sequence(stage_label)

    def start_xy_stage_sequence(self, xy_stage_label: str) -> None:
        '''TODO: test'''
        if not self.is_xy_stage_sequenceable(xy_stage_label):
            raise ValueError(f'XY stage sequence for {xy_stage_label} is not supported')

        self._CORE.start_xy_stage_sequence(xy_stage_label)

    def stderr_log_enabled(self) -> bool:
        return self._CORE.stderr_log_enabled()

    def stop(self, xy_or_z_stage_label: str) -> None:
        type = self.get_device_type(xy_or_z_stage_label)
        if type not in [DeviceType.StageDevice, DeviceType.XYStageDevice]:
            raise ValueError(f'Device "{xy_or_z_stage_label}" is not a stage device')

        self._CORE.stop(xy_or_z_stage_label)

    def stop_exposure_sequence(self, camera_label: str) -> None:
        '''TODO: test'''
        if not self.is_exposure_sequenceable(camera_label):
            raise ValueError(
                f'Exposure sequence is not supported for camera {camera_label}'
            )

        self._CORE.stop_exposure_sequence(camera_label)

    def stop_property_sequence(self, label: str, prop_name: str) -> None:
        '''TODO: test'''
        if not self.is_property_sequenceable(label, prop_name):
            raise ValueError(
                f'Property sequence for {label} property {prop_name} is not supported'
            )

        self._CORE.stop_property_sequence(label, prop_name)

    def stop_secondary_log_file(self, handle: int) -> None:
        self._CORE.stop_secondary_log_file(handle)

    def stop_sequence_acquisition(self, camera_label: Optional[str] = None) -> None:
        '''TODO: test'''
        if camera_label is None:
            self._CORE.stop_sequence_acquisition()
        else:
            self._CORE.stop_sequence_acquisition(camera_label)

    def stop_slm_sequence(self, slm_label: str) -> None:
        '''TODO: test'''
        self._CORE.stop_slm_sequence(slm_label)

    def stop_stage_sequence(self, stage_label: str) -> None:
        '''TODO: test'''
        if not self.is_stage_sequenceable(stage_label):
            raise ValueError(f'Stage sequence for {stage_label} is not supported')

        self._CORE.stop_stage_sequence(stage_label)

    def stop_xy_stage_sequence(self, xy_stage_label: str) -> None:
        '''TODO: test'''
        if not self.is_xy_stage_sequenceable(xy_stage_label):
            raise ValueError(f'XY stage sequence for {xy_stage_label} is not supported')

        self._CORE.stop_xy_stage_sequence(xy_stage_label)

    def supports_device_detection(self, device_label: str) -> bool:
        return self._CORE.supports_device_detection(device_label)

    def system_busy(self) -> bool:
        return self._CORE.system_busy()

    def unload_all_devices(self) -> None:
        self._CORE.unload_all_devices()

    def unload_device(self, label: str) -> None:
        self._CORE.unload_device(label)

    def unload_library(self, module_name: str) -> None:
        self._CORE.unload_library(module_name)

    def update_core_properties(self) -> None:
        self._CORE.update_core_properties()

    def update_system_state_cache(self) -> None:
        self._CORE.update_system_state_cache()

    def uses_device_delay(self, label: str) -> bool:
        return self._CORE.uses_device_delay(label)

    def wait_for_config(self, group: str, config_name: str) -> None:
        self._CORE.wait_for_config(group, config_name)

    def wait_for_device(self, label: str) -> None:
        self._CORE.wait_for_device(label)

    def wait_for_device_type(self, dev_type: Any) -> None:
        '''TODO: requires mmcorej.DeviceType...'''
        self._CORE.wait_for_device_type(dev_type)

    # def wait_for_image_synchro(self) -> None:
    #     self._CORE.wait_for_image_synchro()

    def wait_for_system(self) -> None:
        self._CORE.wait_for_system()

    # def write_to_serial_port(self, port_label: str, data: bytes) -> None:
    #     self._CORE.write_to_serial_port(port_label, data)


if __name__ == '__main__':
    PycroCore()
