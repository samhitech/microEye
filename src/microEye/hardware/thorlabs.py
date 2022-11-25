import numpy as np
import os.path
import logging

from ctypes import *


class IS_SIZE_2D(Structure):
    _fields_ = [('s32Width', c_int), ('s32Height', c_int)]


class IS_POINT_2D(Structure):
    _fields_ = [('s32X', c_int), ('s32Y', c_int)]


class IS_RECT(Structure):
    _pack_ = 8
    _fields_ = [
        ("s32X", c_int),
        ("s32Y", c_int),
        ("s32Width", c_int),
        ("s32Height", c_int),
    ]


class UC480_CAMERA_INFO(Structure):
    _pack_ = 8
    _fields_ = [
        ("dwCameraID", c_uint),
        ("dwDeviceID", c_uint),
        ("dwSensorID", c_uint),
        ("dwInUse", c_uint),
        ("SerNo", (c_char * 16)),
        ("Model", (c_char * 16)),
        ("dwStatus", c_uint),
        ("dwReserved", (c_uint * 2)),
        ("FullModelName", (c_char * 32)),
        ("dwReserved2", (c_uint * 5)),
    ]


class BOARDINFO(Structure):
    _pack_ = 8
    _fields_ = [
        ("SerNo", (c_char * 12)),
        ("ID", (c_char * 20)),
        ("Version", (c_char * 10)),
        ("Date", (c_char * 12)),
        ("Select", c_ubyte),
        ("Type", c_ubyte),
        ("Reserved", (c_char * 8)),
    ]


class SENSORINFO(Structure):
    _pack_ = 8
    _fields_ = [
        ("SensorID", c_ushort),
        ("strSensorName", (c_char * 32)),
        ("nColorMode", c_char),
        ("nMaxWidth", c_uint),
        ("nMaxHeight", c_uint),
        ("bMasterGain", c_int),
        ("bRGain", c_int),
        ("bGGain", c_int),
        ("bBGain", c_int),
        ("bGlobShutter", c_int),
        ("wPixelSize", c_ushort),
        ("nUpperLeftBayerPixel", c_char),
        ("Reserved", (c_char * 13)),
    ]


class IS_DEVICE_INFO_HEARTBEAT(Structure):
    _pack_ = 1
    _fields_ = [
        ("reserved_1", (c_ubyte * 24)),
        ("dwRuntimeFirmwareVersion", c_uint),
        ("reserved_2", (c_ubyte * 8)),
        ("wTemperature", c_ushort),
        ("wLinkSpeed_Mb", c_ushort),
        ("reserved_3", (c_ubyte * 6)),
        ("wComportOffset", c_ushort),
        ("reserved", (c_ubyte * 200)),
    ]


class IS_DEVICE_INFO_CONTROL(Structure):
    _pack_ = 1
    _fields_ = [
        ("dwDeviceId", c_uint),
        ("reserved", (c_ubyte * 148)),
    ]


class IS_DEVICE_INFO(Structure):
    _pack_ = 1
    _fields_ = [
        ("infoDevHeartbeat", IS_DEVICE_INFO_HEARTBEAT),
        ("infoDevControl", IS_DEVICE_INFO_CONTROL),
        ("reserved", (c_ubyte * 240)),
    ]


class UC480_CAPTURE_STATUS_INFO(Structure):
    _pack_ = 8
    _fields_ = [
        ("dwCapStatusCnt_Total", c_uint),
        ("reserved", (c_ubyte * 60)),
        ("adwCapStatusCnt_Detail", (c_uint * 256)),
    ]


class _UC480_CAPTURE_STATUS:
    IS_CAP_STATUS_API_NO_DEST_MEM = 0xa2
    IS_CAP_STATUS_API_CONVERSION_FAILED = 0xa3
    IS_CAP_STATUS_API_IMAGE_LOCKED = 0xa5
    IS_CAP_STATUS_DRV_OUT_OF_BUFFERS = 0xb2
    IS_CAP_STATUS_DRV_DEVICE_NOT_READY = 0xb4
    IS_CAP_STATUS_USB_TRANSFER_FAILED = 0xc7
    IS_CAP_STATUS_DEV_MISSED_IMAGES = 0xe5
    IS_CAP_STATUS_DEV_TIMEOUT = 0xd6
    IS_CAP_STATUS_DEV_FRAME_CAPTURE_FAILED = 0xd9
    IS_CAP_STATUS_ETH_BUFFER_OVERRUN = 0xe4
    IS_CAP_STATUS_ETH_MISSED_IMAGES = 0xe5


def UC480_CAMERA_LIST(uci=(UC480_CAMERA_INFO * 1)):
    _uci = uci if isinstance(uci, type) else type(uci)

    class UC480_CAMERA_LIST(Structure):
        _pack_ = 8
        _fields_ = [
            ("dwCount", c_uint),
            ("uci", _uci),
        ]

    uc480_camera_list = UC480_CAMERA_LIST()

    return uc480_camera_list


class CMD:
    IS_SUCCESS = 0
    IS_NO_SUCCESS = -1
    IS_INVALID_CAMERA_HANDLE = 1
    IS_INVALID_HANDLE = 1
    IS_IO_REQUEST_FAILED = 2
    IS_INVALID_MEMORY_POINTER = 49
    IS_INVALID_PARAMETER = 125


class PCLK_CMD:
    IS_PIXELCLOCK_CMD_GET_NUMBER = 1
    IS_PIXELCLOCK_CMD_GET_LIST = 2
    IS_PIXELCLOCK_CMD_GET_RANGE = 3
    IS_PIXELCLOCK_CMD_GET_DEFAULT = 4
    IS_PIXELCLOCK_CMD_GET = 5
    IS_PIXELCLOCK_CMD_SET = 6


class CS_CMD:
    IS_CAPTURE_STATUS_INFO_CMD_RESET = 1
    IS_CAPTURE_STATUS_INFO_CMD_GET = 2


class _AOI:
    IS_AOI_IMAGE_SET_AOI = 0x0001
    IS_AOI_IMAGE_GET_AOI = 0x0002
    IS_AOI_IMAGE_SET_POS = 0x0003
    IS_AOI_IMAGE_GET_POS = 0x0004
    IS_AOI_IMAGE_SET_SIZE = 0x0005
    IS_AOI_IMAGE_GET_SIZE = 0x0006
    IS_AOI_IMAGE_GET_POS_MIN = 0x0007
    IS_AOI_IMAGE_GET_SIZE_MIN = 0x0008
    IS_AOI_IMAGE_GET_POS_MAX = 0x0009
    IS_AOI_IMAGE_GET_SIZE_MAX = 0x0010
    IS_AOI_IMAGE_GET_POS_INC = 0x0011
    IS_AOI_IMAGE_GET_SIZE_INC = 0x0012
    IS_AOI_IMAGE_GET_POS_X_ABS = 0x0013
    IS_AOI_IMAGE_GET_POS_Y_ABS = 0x0014
    IS_AOI_IMAGE_GET_ORIGINAL_AOI = 0x0015


class _EXP:
    IS_EXPOSURE_CMD_GET_CAPS = 1
    IS_EXPOSURE_CMD_GET_EXPOSURE_DEFAULT = 2
    IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN = 3
    IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX = 4
    IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC = 5
    IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE = 6
    IS_EXPOSURE_CMD_GET_EXPOSURE = 7
    IS_EXPOSURE_CMD_GET_FINE_INCREMENT_RANGE_MIN = 8
    IS_EXPOSURE_CMD_GET_FINE_INCREMENT_RANGE_MAX = 9
    IS_EXPOSURE_CMD_GET_FINE_INCREMENT_RANGE_INC = 10
    IS_EXPOSURE_CMD_GET_FINE_INCREMENT_RANGE = 11
    IS_EXPOSURE_CMD_SET_EXPOSURE = 12
    IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_RANGE_MIN = 13
    IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_RANGE_MAX = 14
    IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_RANGE_INC = 15
    IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_RANGE = 16
    IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_ENABLE = 17
    IS_EXPOSURE_CMD_SET_LONG_EXPOSURE_ENABLE = 18
    IS_EXPOSURE_CMD_GET_DUAL_EXPOSURE_RATIO_DEFAULT = 19
    IS_EXPOSURE_CMD_GET_DUAL_EXPOSURE_RATIO_RANGE = 20
    IS_EXPOSURE_CMD_GET_DUAL_EXPOSURE_RATIO = 21
    IS_EXPOSURE_CMD_SET_DUAL_EXPOSURE_RATIO = 22


class IO_FLASH_PARAMS(Structure):
    _pack_ = 8
    _fields_ = [
        ("s32Delay", c_int),
        ("u32Duration", c_uint),
    ]


class FLASH_MODE:
    IO_FLASH_MODE_OFF = 0
    IO_FLASH_MODE_TRIGGER_LO_ACTIVE = 1
    IO_FLASH_MODE_TRIGGER_HI_ACTIVE = 2
    IO_FLASH_MODE_CONSTANT_HIGH = 3
    IO_FLASH_MODE_CONSTANT_LOW = 4
    IO_FLASH_MODE_FREERUN_LO_ACTIVE = 5
    IO_FLASH_MODE_FREERUN_HI_ACTIVE = 6

    FLASH_MODES = {
        "Flash Off": IO_FLASH_MODE_OFF,
        "Flash Trigger Low Active": IO_FLASH_MODE_TRIGGER_LO_ACTIVE,
        "Flash Trigger High Active": IO_FLASH_MODE_TRIGGER_HI_ACTIVE,
        "Flash Constant High": IO_FLASH_MODE_CONSTANT_HIGH,
        "Flash Constant Low": IO_FLASH_MODE_CONSTANT_LOW,
        "Flash Freerun Low Active": IO_FLASH_MODE_FREERUN_LO_ACTIVE,
        "Flash Freerun High Active": IO_FLASH_MODE_FREERUN_HI_ACTIVE}
    '''Flash modes supported by Thorlabs cameras.

    Returns
    -------
    dict[str, int]
        dictionary used for GUI display and control.
    '''


class DEV_FE_CMD:
    IS_DEVICE_FEATURE_CMD_GET_SUPPORTED_FEATURES = 1
    IS_DEVICE_FEATURE_CMD_SET_LINESCAN_MODE = 2
    IS_DEVICE_FEATURE_CMD_GET_LINESCAN_MODE = 3
    IS_DEVICE_FEATURE_CMD_SET_LINESCAN_NUMBER = 4
    IS_DEVICE_FEATURE_CMD_GET_LINESCAN_NUMBER = 5
    IS_DEVICE_FEATURE_CMD_SET_SHUTTER_MODE = 6
    IS_DEVICE_FEATURE_CMD_GET_SHUTTER_MODE = 7
    IS_DEVICE_FEATURE_CMD_SET_PREFER_XS_HS_MODE = 8
    IS_DEVICE_FEATURE_CMD_GET_PREFER_XS_HS_MODE = 9
    IS_DEVICE_FEATURE_CMD_GET_DEFAULT_PREFER_XS_HS_MODE = 10
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_DEFAULT = 11
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE = 12
    IS_DEVICE_FEATURE_CMD_SET_LOG_MODE = 13
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_VALUE_DEFAULT = 14
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_VALUE_RANGE = 15
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_VALUE = 16
    IS_DEVICE_FEATURE_CMD_SET_LOG_MODE_MANUAL_VALUE = 17
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_GAIN_DEFAULT = 18
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_GAIN_RANGE = 19
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_GAIN = 20
    IS_DEVICE_FEATURE_CMD_SET_LOG_MODE_MANUAL_GAIN = 21
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_MODE_DEFAULT = 22
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_MODE = 23
    IS_DEVICE_FEATURE_CMD_SET_VERTICAL_AOI_MERGE_MODE = 24
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_POSITION_DEFAULT = 25
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_POSITION_RANGE = 26
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_POSITION = 27
    IS_DEVICE_FEATURE_CMD_SET_VERTICAL_AOI_MERGE_POSITION = 28
    IS_DEVICE_FEATURE_CMD_GET_FPN_CORRECTION_MODE_DEFAULT = 29
    IS_DEVICE_FEATURE_CMD_GET_FPN_CORRECTION_MODE = 30
    IS_DEVICE_FEATURE_CMD_SET_FPN_CORRECTION_MODE = 31
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_SOURCE_GAIN_RANGE = 32
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_SOURCE_GAIN_DEFAULT = 33
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_SOURCE_GAIN = 34
    IS_DEVICE_FEATURE_CMD_SET_SENSOR_SOURCE_GAIN = 35
    IS_DEVICE_FEATURE_CMD_GET_BLACK_REFERENCE_MODE_DEFAULT = 36
    IS_DEVICE_FEATURE_CMD_GET_BLACK_REFERENCE_MODE = 37
    IS_DEVICE_FEATURE_CMD_SET_BLACK_REFERENCE_MODE = 38
    IS_DEVICE_FEATURE_CMD_GET_ALLOW_RAW_WITH_LUT = 39
    IS_DEVICE_FEATURE_CMD_SET_ALLOW_RAW_WITH_LUT = 40
    IS_DEVICE_FEATURE_CMD_GET_SUPPORTED_SENSOR_BIT_DEPTHS = 41
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_BIT_DEPTH_DEFAULT = 42
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_BIT_DEPTH = 43
    IS_DEVICE_FEATURE_CMD_SET_SENSOR_BIT_DEPTH = 44
    IS_DEVICE_FEATURE_CMD_GET_TEMPERATURE = 45
    IS_DEVICE_FEATURE_CMD_GET_JPEG_COMPRESSION = 46
    IS_DEVICE_FEATURE_CMD_SET_JPEG_COMPRESSION = 47
    IS_DEVICE_FEATURE_CMD_GET_JPEG_COMPRESSION_DEFAULT = 48
    IS_DEVICE_FEATURE_CMD_GET_JPEG_COMPRESSION_RANGE = 49
    IS_DEVICE_FEATURE_CMD_GET_NOISE_REDUCTION_MODE = 50
    IS_DEVICE_FEATURE_CMD_SET_NOISE_REDUCTION_MODE = 51
    IS_DEVICE_FEATURE_CMD_GET_NOISE_REDUCTION_MODE_DEFAULT = 52
    IS_DEVICE_FEATURE_CMD_GET_TIMESTAMP_CONFIGURATION = 53
    IS_DEVICE_FEATURE_CMD_SET_TIMESTAMP_CONFIGURATION = 54
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_HEIGHT_DEFAULT = 55
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_HEIGHT_NUMBER = 56
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_HEIGHT_LIST = 57
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_HEIGHT = 58
    IS_DEVICE_FEATURE_CMD_SET_VERTICAL_AOI_MERGE_HEIGHT = 59
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_ADDITIONAL_POSITION_DEF = 60
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_ADDITIONAL_POSITION_RANGE = 61
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_ADDITIONAL_POSITION = 62
    IS_DEVICE_FEATURE_CMD_SET_VERTICAL_AOI_MERGE_ADDITIONAL_POSITION = 63
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_TEMPERATURE_NUMERICAL_VALUE = 64
    IS_DEVICE_FEATURE_CMD_SET_IMAGE_EFFECT = 65
    IS_DEVICE_FEATURE_CMD_GET_IMAGE_EFFECT = 66
    IS_DEVICE_FEATURE_CMD_GET_IMAGE_EFFECT_DEFAULT = 67
    IS_DEVICE_FEATURE_CMD_GET_EXTENDED_PIXELCLOCK_RANGE_ENABLE_DEFAULT = 68
    IS_DEVICE_FEATURE_CMD_GET_EXTENDED_PIXELCLOCK_RANGE_ENABLE = 69
    IS_DEVICE_FEATURE_CMD_SET_EXTENDED_PIXELCLOCK_RANGE_ENABLE = 70
    IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_GET_SCOPE = 71
    IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_GET_PARAMS = 72
    IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_SET_PARAMS = 73
    IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_GET_MODE_DEFAULT = 74
    IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_GET_MODE = 75
    IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_SET_MODE = 76
    IS_DEVICE_FEATURE_CMD_SET_I2C_TARGET = 77
    IS_DEVICE_FEATURE_CMD_SET_WIDE_DYNAMIC_RANGE_MODE = 78
    IS_DEVICE_FEATURE_CMD_GET_WIDE_DYNAMIC_RANGE_MODE = 79
    IS_DEVICE_FEATURE_CMD_GET_WIDE_DYNAMIC_RANGE_MODE_DEFAULT = 80
    IS_DEVICE_FEATURE_CMD_GET_SUPPORTED_BLACK_REFERENCE_MODES = 81
    IS_DEVICE_FEATURE_CMD_SET_LEVEL_CONTROLLED_TRIGGER_INPUT_MODE = 82
    IS_DEVICE_FEATURE_CMD_GET_LEVEL_CONTROLLED_TRIGGER_INPUT_MODE = 83
    IS_DEVICE_FEATURE_CMD_GET_LEVEL_CONTROLLED_TRIGGER_INPUT_MODE_DEFAULT = 84
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_MODE_SUPPORTED_LINE_MODES = 85
    IS_DEVICE_FEATURE_CMD_SET_REPEATED_START_CONDITION_I2C = 86
    IS_DEVICE_FEATURE_CMD_GET_REPEATED_START_CONDITION_I2C = 87
    IS_DEVICE_FEATURE_CMD_GET_REPEATED_START_CONDITION_I2C_DEFAULT = 88
    IS_DEVICE_FEATURE_CMD_GET_TEMPERATURE_STATUS = 89
    IS_DEVICE_FEATURE_CMD_GET_MEMORY_MODE_ENABLE = 90
    IS_DEVICE_FEATURE_CMD_SET_MEMORY_MODE_ENABLE = 91
    IS_DEVICE_FEATURE_CMD_GET_MEMORY_MODE_ENABLE_DEFAULT = 92
    IS_DEVICE_FEATURE_CMD_93 = 93
    IS_DEVICE_FEATURE_CMD_94 = 94
    IS_DEVICE_FEATURE_CMD_95 = 95
    IS_DEVICE_FEATURE_CMD_96 = 96
    IS_DEVICE_FEATURE_CMD_GET_SUPPORTED_EXTERNAL_INTERFACES = 97
    IS_DEVICE_FEATURE_CMD_GET_EXTERNAL_INTERFACE = 98
    IS_DEVICE_FEATURE_CMD_SET_EXTERNAL_INTERFACE = 99
    IS_DEVICE_FEATURE_CMD_EXTENDED_AWB_LIMITS_GET = 100
    IS_DEVICE_FEATURE_CMD_EXTENDED_AWB_LIMITS_SET = 101
    IS_DEVICE_FEATURE_CMD_GET_MEMORY_MODE_ENABLE_SUPPORTED = 102


class SENSOR_BIT_DEPTH:
    IS_SENSOR_BIT_DEPTH_AUTO = 0x00000000
    IS_SENSOR_BIT_DEPTH_8_BIT = 0x00000001
    IS_SENSOR_BIT_DEPTH_10_BIT = 0x00000002
    IS_SENSOR_BIT_DEPTH_12_BIT = 0x00000004


class TRIGGER:

    IS_GET_EXTERNALTRIGGER = 0x8000
    IS_SET_TRIGGER_MASK = 0x0100
    IS_SET_TRIGGER_CONTINUOUS = 0x1000
    IS_SET_TRIGGER_OFF = 0x0000
    IS_SET_TRIGGER_HI_LO = (IS_SET_TRIGGER_CONTINUOUS | 0x0001)
    IS_SET_TRIGGER_LO_HI = (IS_SET_TRIGGER_CONTINUOUS | 0x0002)
    IS_SET_TRIGGER_SOFTWARE = (IS_SET_TRIGGER_CONTINUOUS | 0x0008)
    IS_SET_TRIGGER_HI_LO_SYNC = 0x0010
    IS_SET_TRIGGER_LO_HI_SYNC = 0x0020
    IS_SET_TRIGGER_PRE_HI_LO = (IS_SET_TRIGGER_CONTINUOUS | 0x0040)
    IS_SET_TRIGGER_PRE_LO_HI = (IS_SET_TRIGGER_CONTINUOUS | 0x0080)

    TRIGGER_MODES = {
        "Trigger Off": IS_SET_TRIGGER_OFF,
        "Software Trigger": IS_SET_TRIGGER_SOFTWARE,
        "Falling edge external trigger": IS_SET_TRIGGER_HI_LO,
        "Rising edge external trigger": IS_SET_TRIGGER_LO_HI}
    '''Tigger modes supported by Thorlabs cameras.

    Returns
    -------
    dict[str, int]
        dictionary used for GUI display and control.
    '''


class IO_CMD:
    IS_IO_CMD_GPIOS_GET_SUPPORTED = 1
    IS_IO_CMD_GPIOS_GET_SUPPORTED_INPUTS = 2
    IS_IO_CMD_GPIOS_GET_SUPPORTED_OUTPUTS = 3
    IS_IO_CMD_GPIOS_GET_DIRECTION = 4
    IS_IO_CMD_GPIOS_SET_DIRECTION = 5
    IS_IO_CMD_GPIOS_GET_STATE = 6
    IS_IO_CMD_GPIOS_SET_STATE = 7
    IS_IO_CMD_LED_GET_STATE = 8
    IS_IO_CMD_LED_SET_STATE = 9
    IS_IO_CMD_LED_TOGGLE_STATE = 10
    IS_IO_CMD_FLASH_GET_GLOBAL_PARAMS = 11
    IS_IO_CMD_FLASH_APPLY_GLOBAL_PARAMS = 12
    IS_IO_CMD_FLASH_GET_SUPPORTED_GPIOS = 13
    IS_IO_CMD_FLASH_GET_PARAMS_MIN = 14
    IS_IO_CMD_FLASH_GET_PARAMS_MAX = 15
    IS_IO_CMD_FLASH_GET_PARAMS_INC = 16
    IS_IO_CMD_FLASH_GET_PARAMS = 17
    IS_IO_CMD_FLASH_SET_PARAMS = 18
    IS_IO_CMD_FLASH_GET_MODE = 19
    IS_IO_CMD_FLASH_SET_MODE = 20
    IS_IO_CMD_PWM_GET_SUPPORTED_GPIOS = 21
    IS_IO_CMD_PWM_GET_PARAMS_MIN = 22
    IS_IO_CMD_PWM_GET_PARAMS_MAX = 23
    IS_IO_CMD_PWM_GET_PARAMS_INC = 24
    IS_IO_CMD_PWM_GET_PARAMS = 25
    IS_IO_CMD_PWM_SET_PARAMS = 26
    IS_IO_CMD_PWM_GET_MODE = 27
    IS_IO_CMD_PWM_SET_MODE = 28
    IS_IO_CMD_GPIOS_GET_CONFIGURATION = 29
    IS_IO_CMD_GPIOS_SET_CONFIGURATION = 30
    IS_IO_CMD_FLASH_GET_GPIO_PARAMS_MIN = 31
    IS_IO_CMD_FLASH_SET_GPIO_PARAMS = 32
    IS_IO_CMD_FLASH_GET_AUTO_FREERUN_DEFAULT = 33
    IS_IO_CMD_FLASH_GET_AUTO_FREERUN = 34
    IS_IO_CMD_FLASH_SET_AUTO_FREERUN = 35


class ColorMode:
    IS_COLORMODE_INVALID = 0
    IS_COLORMODE_MONOCHROME = 1
    IS_COLORMODE_BAYER = 2
    IS_COLORMODE_CBYCRY = 4
    IS_COLORMODE_JPEG = 8


ColorModeStr = {
    ColorMode.IS_COLORMODE_INVALID: 'IS_COLORMODE_INVALID',
    ColorMode.IS_COLORMODE_MONOCHROME: 'IS_COLORMODE_MONOCHROME',
    ColorMode.IS_COLORMODE_BAYER: 'IS_COLORMODE_BAYER',
    ColorMode.IS_COLORMODE_CBYCRY: 'IS_COLORMODE_CBYCRY',
    ColorMode.IS_COLORMODE_JPEG: 'IS_COLORMODE_JPEG'}


class _ColorModes:
    IS_CM_SENSOR_RAW8 = 11
    IS_CM_SENSOR_RAW10 = 33
    IS_CM_SENSOR_RAW12 = 27
    IS_CM_SENSOR_RAW16 = 29
    IS_CM_MONO8 = 6
    IS_CM_MONO10 = 34
    IS_CM_MONO12 = 26
    IS_CM_MONO16 = 28

    IS_CM_ORDER_BGR = 0x0000
    IS_CM_ORDER_RGB = 0x0080
    IS_CM_ORDER_MASK = 0x0080
    IS_CM_FORMAT_PLANAR = 0x2000
    IS_CM_BGR5_PACKED = (3 | IS_CM_ORDER_BGR)
    IS_CM_BGR565_PACKED = (2 | IS_CM_ORDER_BGR)
    IS_CM_RGB8_PACKED = (1 | IS_CM_ORDER_RGB)
    IS_CM_BGR8_PACKED = (1 | IS_CM_ORDER_BGR)
    IS_CM_RGBA8_PACKED = (0 | IS_CM_ORDER_RGB)
    IS_CM_BGRA8_PACKED = (0 | IS_CM_ORDER_BGR)
    IS_CM_RGBY8_PACKED = (24 | IS_CM_ORDER_RGB)
    IS_CM_BGRY8_PACKED = (24 | IS_CM_ORDER_BGR)
    IS_CM_RGB10_PACKED = (25 | IS_CM_ORDER_RGB)
    IS_CM_BGR10_PACKED = (25 | IS_CM_ORDER_BGR)
    IS_CM_RGB10_UNPACKED = (35 | IS_CM_ORDER_RGB)
    IS_CM_BGR10_UNPACKED = (35 | IS_CM_ORDER_BGR)
    IS_CM_RGB12_UNPACKED = (30 | IS_CM_ORDER_RGB)
    IS_CM_BGR12_UNPACKED = (30 | IS_CM_ORDER_BGR)
    IS_CM_RGBA12_UNPACKED = (31 | IS_CM_ORDER_RGB)
    IS_CM_BGRA12_UNPACKED = (31 | IS_CM_ORDER_BGR)
    IS_CM_UYVY_PACKED = 12
    IS_CM_UYVY_MONO_PACKED = 13
    IS_CM_UYVY_BAYER_PACKED = 14
    IS_CM_CBYCRY_PACKED = 23
    IS_CM_RGB8_PLANAR = (1 | IS_CM_ORDER_RGB | IS_CM_FORMAT_PLANAR)


formats_ = {_ColorModes.IS_CM_SENSOR_RAW8: 8,
            _ColorModes.IS_CM_SENSOR_RAW10: 16,
            _ColorModes.IS_CM_SENSOR_RAW12: 16,
            _ColorModes.IS_CM_SENSOR_RAW16: 16,

            _ColorModes.IS_CM_MONO8: 8,
            _ColorModes.IS_CM_MONO10: 16,
            _ColorModes.IS_CM_MONO12: 16,
            _ColorModes.IS_CM_MONO16: 16,

            _ColorModes.IS_CM_RGB8_PLANAR: 24,
            _ColorModes.IS_CM_RGB8_PACKED: 24,
            _ColorModes.IS_CM_RGBA8_PACKED: 32,
            _ColorModes.IS_CM_RGBY8_PACKED: 32,
            _ColorModes.IS_CM_RGB10_PACKED: 32,

            _ColorModes.IS_CM_RGB10_UNPACKED: 48,
            _ColorModes.IS_CM_RGB12_UNPACKED: 48,
            _ColorModes.IS_CM_RGBA12_UNPACKED: 64,

            _ColorModes.IS_CM_BGR5_PACKED: 16,
            _ColorModes.IS_CM_BGR565_PACKED: 16,
            _ColorModes.IS_CM_BGR8_PACKED: 24,
            _ColorModes.IS_CM_BGRA8_PACKED: 32,
            _ColorModes.IS_CM_BGRY8_PACKED: 32,
            _ColorModes.IS_CM_BGR10_PACKED: 32,

            _ColorModes.IS_CM_BGR10_UNPACKED: 48,
            _ColorModes.IS_CM_BGR12_UNPACKED: 48,
            _ColorModes.IS_CM_BGRA12_UNPACKED: 64,

            _ColorModes.IS_CM_UYVY_PACKED: 16,
            _ColorModes.IS_CM_UYVY_MONO_PACKED: 16,
            _ColorModes.IS_CM_UYVY_BAYER_PACKED: 16,
            _ColorModes.IS_CM_CBYCRY_PACKED: 16
            }

formats_strs = {_ColorModes.IS_CM_SENSOR_RAW8: "IS_CM_SENSOR_RAW8",
                _ColorModes.IS_CM_SENSOR_RAW10: "IS_CM_SENSOR_RAW10",
                _ColorModes.IS_CM_SENSOR_RAW12: "IS_CM_SENSOR_RAW12",
                _ColorModes.IS_CM_SENSOR_RAW16: "IS_CM_SENSOR_RAW16",

                _ColorModes.IS_CM_MONO8: "IS_CM_MONO8",
                _ColorModes.IS_CM_MONO10: "IS_CM_MONO10",
                _ColorModes.IS_CM_MONO12: "IS_CM_MONO12",
                _ColorModes.IS_CM_MONO16: "IS_CM_MONO16",

                _ColorModes.IS_CM_RGB8_PLANAR: "IS_CM_RGB8_PLANAR",
                _ColorModes.IS_CM_RGB8_PACKED: "IS_CM_RGB8_PACKED",
                _ColorModes.IS_CM_RGBA8_PACKED: "IS_CM_RGB8_PACKED",
                _ColorModes.IS_CM_RGBY8_PACKED: "IS_CM_RGBY8_PACKED",
                _ColorModes.IS_CM_RGB10_PACKED: "IS_CM_RGB10_PACKED",

                _ColorModes.IS_CM_RGB10_UNPACKED: "IS_CM_RGB10_UNPACKED",
                _ColorModes.IS_CM_RGB12_UNPACKED: "IS_CM_RGB12_UNPACKED",
                _ColorModes.IS_CM_RGBA12_UNPACKED: "IS_CM_RGBA12_UNPACKED",

                _ColorModes.IS_CM_BGR5_PACKED: "IS_CM_BGR5_PACKED",
                _ColorModes.IS_CM_BGR565_PACKED: "IS_CM_BGR565_PACKED",
                _ColorModes.IS_CM_BGR8_PACKED: "IS_CM_BGR8_PACKED",
                _ColorModes.IS_CM_BGRA8_PACKED: "IS_CM_BGRA8_PACKED",
                _ColorModes.IS_CM_BGRY8_PACKED: "IS_CM_BGRY8_PACKED",
                _ColorModes.IS_CM_BGR10_PACKED: "IS_CM_BGR10_PACKED",

                _ColorModes.IS_CM_BGR10_UNPACKED: "IS_CM_BGR10_UNPACKED",
                _ColorModes.IS_CM_BGR12_UNPACKED: "IS_CM_BGR12_UNPACKED",
                _ColorModes.IS_CM_BGRA12_UNPACKED: "IS_CM_BGRA12_UNPACKED",

                _ColorModes.IS_CM_UYVY_PACKED: "IS_CM_UYVY_PACKED",
                _ColorModes.IS_CM_UYVY_MONO_PACKED: "IS_CM_UYVY_MONO_PACKED",
                _ColorModes.IS_CM_UYVY_BAYER_PACKED: "IS_CM_UYVY_BAYER_PACKED",
                _ColorModes.IS_CM_CBYCRY_PACKED: "IS_CM_CBYCRY_PACKED"
                }


def get_data(image_mem, x, y, bits, pitch, copy):
    data = None
    if copy:
        mem = create_string_buffer(y * pitch)
        memmove(mem, image_mem, y * pitch)
        data = np.frombuffer(mem, dtype=np.uint8) if np else mem
    else:
        data = np.ctypeslib.as_array(
            cast(image_mem, POINTER(c_ubyte)),
            (y * pitch, )) if np else image_mem

    return data


IS_DEVICE_INFO_CMD_GET_DEVICE_INFO = 0x02010001
IS_SET_DM_DIB = 1
IS_SET_DM_DIRECT3D = 4
IS_SET_DM_OPENGL = 8
IS_WAIT = 0x0001
IS_DONT_WAIT = 0x0000
IS_FORCE_VIDEO_STOP = 0x4000
IS_FORCE_VIDEO_START = 0x4000
IS_USE_NEXT_MEM = 0x8000
IS_TIMED_OUT = 122
IS_CAPTURE_STATUS = 0x0003
IS_TRIGGER_MISSED = 20
IS_GET_STATUS = 0x8000
IS_GET_COLOR_MODE = 0x8000


class thorlabs_camera():

    uc480_file = 'C:\\Program Files\\Thorlabs\\Scientific' + \
        ' Imaging\\ThorCam\\uc480_64.dll'

    def __init__(self, hCam=0):
        if os.path.isfile(thorlabs_camera.uc480_file):
            self.bit_depth = int(8)
            self.supported_bit_depth = c_uint(0)
            self.bytes_per_pixel = int(self.bit_depth / 8)
            self.color_mode = c_int(0)
            self.minAOI = IS_SIZE_2D()
            self.rectAOI = IS_RECT()
            self.set_rectAOI = IS_RECT()
            self.cam = None
            self.hCam = c_int(hCam)
            self.meminfo = None
            self.exposure_current = c_double(0.05)
            self.exposure_range = (c_double * 3)()
            self.roi_pos = None
            self.minFrameRate = c_double(0)
            self.maxFrameRate = c_double(0)
            self.incFrameRate = c_double(0)
            self.currentFrameRate = c_double(0)
            self.pixel_clock = c_uint(0)
            self.pixel_clock_def = c_uint(0)
            self.pixel_clock_count = c_uint(0)
            self.pixel_clock_list = None

            self.MemInfo = []
            self.pitch = c_int()
            self.current_buffer = c_void_p()
            self.current_id = c_int()

            self.flash_mode = c_uint(FLASH_MODE.IO_FLASH_MODE_OFF)
            self.flash_min = IO_FLASH_PARAMS()
            self.flash_max = IO_FLASH_PARAMS()
            self.flash_inc = IO_FLASH_PARAMS()
            self.flash_cur = IO_FLASH_PARAMS()

            self.sInfo = SENSORINFO()
            self.cInfo = BOARDINFO()
            self.dInfo = IS_DEVICE_INFO()
            self.dInfo_not_supported = False

            self.CaptureStatusInfo = UC480_CAPTURE_STATUS_INFO()

            self.acquisition = False
            self.capture_video = False
            self.memory_allocated = False
            self.trigger_mode = None
            self.temperature = -127
            self.uc480 = windll.LoadLibrary(thorlabs_camera.uc480_file)
        else:
            raise Exception("Please install ThorCam drivers.")

    @staticmethod
    def get_camera_list(output=False):
        '''Gets the list of available IDS cameras

        Parameters
        ----------
        output : bool, optional
            print out camera list, by default False

        Returns
        -------
        dict[str, object]
            dictionary containing
            {camID, devID, senID, Status, InUse, Model, Serial}
        '''
        uc480 = windll.LoadLibrary(thorlabs_camera.uc480_file)
        cam_list = []
        cam_count = c_int(0)
        nRet = uc480.is_GetNumberOfCameras(byref(cam_count))
        if nRet == CMD.IS_SUCCESS and cam_count.value > 0:
            pucl = UC480_CAMERA_LIST(
                UC480_CAMERA_INFO * cam_count.value)
            pucl.dwCount = cam_count.value
            if (uc480.is_GetCameraList(byref(pucl)) == CMD.IS_SUCCESS):
                for index in range(cam_count.value):
                    cam_list.append({
                        "camID": pucl.uci[index].dwCameraID,
                        "devID": pucl.uci[index].dwDeviceID,
                        "senID": pucl.uci[index].dwSensorID,
                        "Status": pucl.uci[index].dwStatus,
                        "InUse": pucl.uci[index].dwInUse,
                        "Model": pucl.uci[index].Model.decode('utf-8'),
                        "Serial": pucl.uci[index].SerNo.decode('utf-8'),
                        "Driver": 'Thorlabs DCx UC480'})
                if output:
                    print(cam_list)
        return cam_list

    def get_coded_info(self):
        '''Reads out the data hard-coded in the non-volatile camera memory
        and writes it to the data structure that cInfo points to
        '''
        nRet = self.uc480.is_GetCameraInfo(self.hCam, byref(self.cInfo))
        if nRet != CMD.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

    def get_device_info(self):
        '''Gets the device info, used to get the sensor temp.
        '''
        if not self.dInfo_not_supported:
            nRet = self.uc480.is_DeviceInfo(
                self.hCam,
                IS_DEVICE_INFO_CMD_GET_DEVICE_INFO,
                byref(self.dInfo),
                sizeof(IS_DEVICE_INFO))

            if nRet != CMD.IS_SUCCESS:
                self.dInfo_not_supported = True
                print("is_DeviceInfo ERROR", nRet)

            return nRet
        else:
            return 125

    def get_temperature(self):
        '''Reads out the sensor temperature value

        Returns
        -------
        float
            camera sensor temperature in C.
        '''
        nRet = self.get_device_info()
        self.temperature = -127
        if nRet == CMD.IS_SUCCESS:
            self.temperature = ((
                (self.dInfo.infoDevHeartbeat.wTemperature >> 4) & 127) * 1.0
                + ((self.dInfo.infoDevHeartbeat.wTemperature & 15) / 10.0))
        return self.temperature

    def get_sensor_info(self):
        '''You can query additional information about
        the sensor type used in the camera
        '''
        nRet = self.uc480.is_GetSensorInfo(self.hCam, byref(self.sInfo))
        if nRet != CMD.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

    def resetToDefault(self):
        '''Resets all parameters to the camera-specific defaults
        as specified by the driver.

        By default, the camera uses full resolution, a medium speed
        and color level gain values adapted to daylight exposure.
        '''
        nRet = self.uc480.is_ResetToDefault(self.hCam)
        if nRet != CMD.IS_SUCCESS:
            print("is_ResetToDefault ERROR")

    def initialize(self):

        is_InitCamera = self.uc480.is_InitCamera
        is_InitCamera.argtypes = [POINTER(c_int)]
        i = is_InitCamera(byref(self.hCam))

        if i == 0:
            print("Camera initialized successfully.")

            self.get_coded_info()
            self.get_sensor_info()
            self.resetToDefault()
            self.setDisplayMode()

            self.setColorMode()
            # self.is_GetColorDepth()

            self.get_AOI()
            self.get_minAOI()

            self.get_pixel_clock_info(True)
            self.set_pixel_clock(self.pixel_clock_def.value)
            self.get_framerate_range()
            self.get_exposure_range()
            self.get_flash_range()
            self.print_cam_info()

            self.name = self.sInfo.strSensorName.decode('utf-8') + "_" + \
                self.cInfo.SerNo.decode('utf-8')
            self.name = str.replace(self.name, "-", "_")

            return CMD.IS_SUCCESS
        else:
            print(
                "Camera initialization failed with error code "+str(i))
            return i

    def refresh_info(self, output=False):
        '''Refreshes the camera's pixel clock,
        framerate, exposure and flash ranges.

        Parameters
        ----------
        output : bool, optional
            True to printout errors, by default False
        '''
        if output:
            print(self.name)
            print()
        self.get_pixel_clock_info(output)
        self.get_framerate_range(output)
        self.get_exposure_range(output)
        self.get_exposure(output)
        self.get_flash_range(output)

    # Prints out some information about the camera and the sensor
    def print_cam_info(self):
        print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))
        print("Maximum image width:\t", self.width)
        print("Maximum image height:\t", self.height)
        print()

    def close(self):
        if self.hCam is not None:
            self.stop_live_capture()
            i = self.uc480.is_ExitCamera(self.hCam)
            if i == 0:
                print("Camera closed successfully.")
            else:
                print("Closing the camera failed with error code "+str(i))
        else:
            return

    def stop_live_capture(self):
        self.uc480.is_StopLiveVideo(self.hCam, 1)

    def setColorMode(self):
        '''Set the right color mode, by camera default.
        '''
        # self.color_mode = self.uc480.is_SetColorMode(
        #     self.hCam, IS_GET_COLOR_MODE)
        nCmode = int.from_bytes(self.sInfo.nColorMode, byteorder='big')
        nRet = self.uc480.is_DeviceFeature(
            self.hCam,
            DEV_FE_CMD.IS_DEVICE_FEATURE_CMD_GET_SUPPORTED_SENSOR_BIT_DEPTHS,
            byref(self.supported_bit_depth), sizeof(self.supported_bit_depth))
        print('nRet', nRet)
        if nCmode == \
                ColorMode.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            self.uc480.is_GetColorDepth(
                self.hCam,
                byref(self.bit_depth),
                byref(self.color_mode))
            self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
            print(ColorModeStr[nCmode], ": ", )
            print("\tcolor_mode: \t\t", formats_strs[self.color_mode])
            print("\tbit_depth: \t\t", self.bit_depth)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif nCmode ==\
                ColorMode.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.color_mode = _ColorModes.IS_CM_BGRA8_PACKED
            self.bit_depth = int(32)
            self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
            print(ColorModeStr[nCmode], ": ", )
            print("\tcolor_mode: \t\t", formats_strs[self.color_mode])
            print("\tbit_depth: \t\t", self.bit_depth)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif nCmode ==\
                ColorMode.IS_COLORMODE_MONOCHROME:
            # for mono camera models that uses CM_MONO12 mode
            print('Sbit', self.supported_bit_depth)
            if nRet == CMD.IS_SUCCESS:
                if (self.supported_bit_depth and
                        SENSOR_BIT_DEPTH.IS_SENSOR_BIT_DEPTH_12_BIT) != 0:
                    self.color_mode = _ColorModes.IS_CM_MONO12
                    self.bit_depth = int(12)
                    self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
                elif (self.supported_bit_depth and
                        SENSOR_BIT_DEPTH.IS_SENSOR_BIT_DEPTH_10_BIT) != 0:
                    self.color_mode = _ColorModes.IS_CM_MONO10
                    self.bit_depth = int(10)
                    self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
                elif (self.supported_bit_depth and
                        SENSOR_BIT_DEPTH.IS_SENSOR_BIT_DEPTH_8_BIT) != 0:
                    self.color_mode = _ColorModes.IS_CM_MONO8
                    self.bit_depth = int(8)
                    self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
            else:
                self.color_mode = _ColorModes.IS_CM_MONO8
                self.bit_depth = int(8)
                self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
            print(ColorModeStr[nCmode], ": ", )
            print("\tcolor_mode: \t\t", formats_strs[self.color_mode])
            print("\tbit_depth: \t\t", self.bit_depth)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            self.color_mode = _ColorModes.IS_CM_MONO8
            self.bit_depth = int(8)
            self.bytes_per_pixel = int(np.ceil(self.bit_depth / 8))
            print("Else: ", )
            print("\tcolor_mode: \t\t", formats_strs[self.color_mode])
            print("\tbit_depth: \t\t", self.bit_depth)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)

        nRet = self.uc480.is_SetColorMode(
            self.hCam, self.color_mode)

    def is_GetColorDepth(self):
        is_GetColorDepth = self.uc480.is_GetColorDepth
        is_GetColorDepth.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
        is_GetColorDepth(
            self.hCam, byref(self.bit_depth), byref(self.color_mode)
        )

        self.bytes_per_pixel = int(self.bit_depth.value / 8)

    def setDisplayMode(self, mode=IS_SET_DM_DIB):
        '''Captures an image in system memory (RAM).

        Using is_RenderBitmap(), you can define the image display (default).

        Parameters
        ----------
        mode : [type], optional
            Capture mode, by default IS_SET_DM_DIB
            captures an image in system memory (RAM).

        Returns
        -------
        int
            is_SetDisplayMode return code
        '''
        nRet = self.uc480.is_SetDisplayMode(self.hCam, mode)
        return nRet

    def get_pixel_clock(self, output=True):
        '''Gets the current pixel clock speed.

        Parameters
        ----------
        output : bool, optional
            print out to terminal, by default True

        Returns
        -------
        int
            pixel clock speed in MHz.
        '''
        is_PixelClock = self.uc480.is_PixelClock
        is_PixelClock.argtypes = [c_int, c_uint, POINTER(c_uint), c_uint]
        nRet = is_PixelClock(
            self.hCam,
            PCLK_CMD.IS_PIXELCLOCK_CMD_GET,
            byref(self.pixel_clock),
            sizeof(self.pixel_clock))

        if nRet != 0:
            print("is_PixelClock ERROR")
        elif output:
            print("Pixel Clock %d MHz" % self.pixel_clock.value)
        return self.pixel_clock.value

    def set_pixel_clock(self, value):
        '''Sets the camera's pixel clock speed.

        Parameters
        ----------
        value : uint
            The supported clock speed to set.

        Returns
        -------
        int
            is_PixelClock return code.
        '''
        set = c_uint(value)
        is_PixelClock = self.uc480.is_PixelClock
        is_PixelClock.argtypes = [c_int, c_uint, POINTER(c_uint), c_uint]
        nRet = is_PixelClock(
            self.hCam,
            PCLK_CMD.IS_PIXELCLOCK_CMD_SET,
            byref(set),
            sizeof(set))

        if nRet != CMD.IS_SUCCESS:
            print("is_PixelClock ERROR")

        return nRet

    def get_pixel_clock_info(self, output=False):
        self.get_pixel_clock()

        is_PixelClock = self.uc480.is_PixelClock
        is_PixelClock.argtypes = [c_int, c_uint, POINTER(c_uint), c_uint]
        nRet = is_PixelClock(
            self.hCam,
            PCLK_CMD.IS_PIXELCLOCK_CMD_GET_DEFAULT,
            byref(self.pixel_clock_def),
            sizeof(self.pixel_clock_def))

        nRet = is_PixelClock(
            self.hCam,
            PCLK_CMD.IS_PIXELCLOCK_CMD_GET_NUMBER,
            byref(self.pixel_clock_count),
            sizeof(self.pixel_clock_count))

        if nRet != 0:
            print("is_PixelClock Count ERROR")
        else:
            if output:
                print("Count " + str(self.pixel_clock_count.value))

            self.pixel_clock_list = (c_uint *
                                     self.pixel_clock_count.value)()

            nRet = is_PixelClock(
                self.hCam,
                PCLK_CMD.IS_PIXELCLOCK_CMD_GET_LIST,
                self.pixel_clock_list,
                self.pixel_clock_count.value * sizeof(c_uint))

            if nRet != 0:
                print("is_PixelClock List ERROR")
            elif output:
                for clk in self.pixel_clock_list:
                    print("List " + str(clk))
                print()

    def get_AOI(self):
        '''Can be used to get the size and position
        of an "area of interest" (AOI) within an image.

        For AOI rectangle check IDS_Camera.rectAOI

        Returns
        -------
        int
            is_AOI return code.
        '''
        nRet = self.uc480.is_AOI(
            self.hCam,
            _AOI.IS_AOI_IMAGE_GET_AOI,
            byref(self.rectAOI),
            sizeof(self.rectAOI))

        if nRet != CMD.IS_SUCCESS:
            print("is_AOI GET ERROR")

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height
        return nRet

    def get_minAOI(self):
        '''Can be used to get the size of the minimum
        "area of interest" (AOI).

        Returns
        -------
        int
            is_AOI return code.
        '''
        nRet = self.uc480.is_AOI(
            self.hCam,
            _AOI.IS_AOI_IMAGE_GET_SIZE_MIN,
            byref(self.minAOI),
            sizeof(self.minAOI))
        if nRet != CMD.IS_SUCCESS:
            print("is_AOI GET MIN SIZE ERROR")
        return nRet

    def set_AOI(self, x, y, width, height):
        '''Sets the size and position of an
        "area of interest"(AOI) within an image.

        Parameters
        ----------
        x : int
            AOI x position.
        y : int
            AOI y position.
        width : int
            AOI width.
        height : int
            AOI height.

        Returns
        -------
        int
            is_AOI return code.
        '''
        self.set_rectAOI.s32X = x
        self.set_rectAOI.s32Y = y
        self.set_rectAOI.s32Width = max(self.minAOI.s32Width, width)
        self.set_rectAOI.s32Height = max(self.minAOI.s32Height, height)

        nRet = self.uc480.is_AOI(
            self.hCam,
            _AOI.IS_AOI_IMAGE_SET_AOI,
            byref(self.set_rectAOI),
            sizeof(self.set_rectAOI))

        if nRet != CMD.IS_SUCCESS:
            print("is_AOI SET ERROR")

        self.width = self.set_rectAOI.s32Width
        self.height = self.set_rectAOI.s32Height

        return nRet

    def reset_AOI(self):
        '''Resets the AOI.

        Returns
        -------
        int
            is_AOI return code.
        '''
        nRet = self.uc480.is_AOI(
            self.hCam,
            _AOI.IS_AOI_IMAGE_SET_AOI,
            byref(self.rectAOI),
            sizeof(self.rectAOI))

        if nRet != CMD.IS_SUCCESS:
            print("is_AOI RESET ERROR")

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height

        return nRet

    # Set exposure
    def set_exposure(self, value):
        is_Exposure = self.uc480.is_Exposure
        is_Exposure.argtypes = [c_int, c_uint, POINTER(c_double), c_uint]
        nRet = is_Exposure(
            self.hCam,
            _EXP.IS_EXPOSURE_CMD_SET_EXPOSURE,
            c_double(max(
                min(value, self.exposure_range[1]), self.exposure_range[0])),
            sizeof(c_double))
        if nRet != CMD.IS_SUCCESS:
            print("is_Exposure Set ERROR")
        else:
            print("Exposure set to " + str(self.get_exposure(False)))
            print()
        return nRet

    def get_exposure_range(self, output=True):
        ''' Gets exposure range

        Parameters
        ----------
        output : bool, optional
            [description], by default True
        '''
        nRet = self.uc480.is_Exposure(
            self.hCam,
            _EXP.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE,
            self.exposure_range, sizeof(self.exposure_range))
        if nRet != CMD.IS_SUCCESS:
            print("is_Exposure Range ERROR")
        elif output:
            print("Exposure")
            print("Min " + str(self.exposure_range[0]))
            print("Max " + str(self.exposure_range[1]))
            print("Inc " + str(self.exposure_range[2]))
            print()

    # Get exposure
    def get_exposure(self, output=True):
        nRet = self.uc480.is_Exposure(
            self.hCam,
            _EXP.IS_EXPOSURE_CMD_GET_EXPOSURE,
            byref(self.exposure_current), sizeof(self.exposure_current))
        if nRet != CMD.IS_SUCCESS:
            print("is_Exposure Current ERROR")
        elif output:
            print("Current Exposure " + str(self.exposure_current.value))
            print()
        return self.exposure_current.value

    # Get framerate range
    def get_framerate_range(self, output=True):
        nRet = self.uc480.is_GetFrameTimeRange(
            self.hCam,
            byref(self.minFrameRate),
            byref(self.maxFrameRate),
            byref(self.incFrameRate))
        if nRet != CMD.IS_SUCCESS:
            print("is_GetFrameTimeRange ERROR")
        else:
            temp = self.maxFrameRate.value
            self.maxFrameRate.value = 1/self.minFrameRate.value
            self.minFrameRate.value = 1/temp
            if output:
                print("FrameRate")
                print("Min " + str(self.minFrameRate.value))
                print("Max " + str(self.maxFrameRate.value))
                print()
        return np.array([self.minFrameRate.value, self.maxFrameRate.value])

    # Get current framerate
    def get_framerate(self, output=True):
        nRet = self.uc480.is_GetFramesPerSecond(
            self.hCam, byref(self.currentFrameRate))
        if nRet != CMD.IS_SUCCESS:
            print("is_GetFramesPerSecond ERROR")
        elif output:
            print("Current FrameRate " + str(self.currentFrameRate.value))
            print()
        return self.currentFrameRate.value

    # Set current framerate
    def set_framerate(self, value):
        nRet = self.uc480.is_SetFrameRate(
            self.hCam,
            c_double(max(
                min(value, self.maxFrameRate.value), self.minFrameRate.value)),
            byref(self.currentFrameRate))
        if nRet != CMD.IS_SUCCESS:
            print("is_SetFrameRate ERROR")
        else:
            print("FrameRate set to " + str(self.currentFrameRate.value))
            print()
        return nRet

    # get trigger mode
    def get_trigger_mode(self):
        nRet = self.uc480.is_SetExternalTrigger(
            self.hCam,
            TRIGGER.IS_GET_EXTERNALTRIGGER)
        if nRet == TRIGGER.IS_SET_TRIGGER_OFF:
            print("Trigger Off")
        elif nRet == TRIGGER.IS_SET_TRIGGER_SOFTWARE:
            print("Software Trigger")
        elif nRet == TRIGGER.IS_SET_TRIGGER_HI_LO:
            print("Falling edge external trigger")
        elif nRet == TRIGGER.IS_SET_TRIGGER_LO_HI:
            print("Rising  edge external trigger")
        else:
            print("NA")
        self.trigger_mode = nRet
        return nRet

    def set_trigger_mode(self, mode):
        return self.uc480.is_SetExternalTrigger(self.hCam, mode)

    # Get flash output info
    def get_flash_range(self, output=True):
        nRet = self.uc480.is_IO(
            self.hCam,
            IO_CMD.IS_IO_CMD_FLASH_GET_PARAMS_MIN,
            byref(self.flash_min),
            sizeof(self.flash_min))
        if nRet != CMD.IS_SUCCESS:
            print("is_IO Flash Min ERROR")
        elif output:
            print("Flash Output")
            print("Min Delay (us) " + str(self.flash_min.s32Delay))
            print("Min Duration (us) " + str(self.flash_min.u32Duration))
            print()

        nRet = self.uc480.is_IO(
            self.hCam,
            IO_CMD.IS_IO_CMD_FLASH_GET_PARAMS_MAX,
            byref(self.flash_max),
            sizeof(self.flash_max))
        if nRet != CMD.IS_SUCCESS:
            print("is_IO Flash Max ERROR")
        elif output:
            print("Max Delay (us) " + str(self.flash_max.s32Delay))
            print("Max Duration (us) " + str(self.flash_max.u32Duration))
            print()

        nRet = self.uc480.is_IO(
            self.hCam,
            IO_CMD.IS_IO_CMD_FLASH_GET_PARAMS_INC,
            byref(self.flash_inc),
            sizeof(self.flash_inc))
        if nRet != CMD.IS_SUCCESS:
            print("is_IO Flash Inc. ERROR")
        elif output:
            print("Inc. Delay (us) " + str(self.flash_inc.s32Delay))
            print(
                "Inc. Duration (us) " + str(self.flash_inc.u32Duration))
            print()

        self.get_flash_params(output)

    def get_flash_params(self, output=True):
        nRet = self.uc480.is_IO(
            self.hCam,
            IO_CMD.IS_IO_CMD_FLASH_GET_PARAMS,
            byref(self.flash_cur),
            sizeof(self.flash_cur))
        if nRet != CMD.IS_SUCCESS:
            print("is_IO Flash Current ERROR")
        elif output:
            print("Current Delay (us) " + str(self.flash_cur.s32Delay))
            print("Current Duration (us) " + str(
                self.flash_cur.u32Duration))
            print()

    # Set current flash parameters
    def set_flash_params(self, delay, duration):
        params = IO_FLASH_PARAMS()
        delay = int(delay)
        duration = c_uint(duration)
        if duration != 0:
            params.u32Duration = max(
                min(duration, self.flash_max.u32Duration),
                self.flash_min.u32Duration)
        else:
            params.u32Duration = duration
        params.s32Delay.value = max(
            min(delay, self.flash_max.s32Delay),
            self.flash_min.s32Delay)
        nRet = self.uc480.is_IO(
            self.hCam,
            IO_CMD.IS_IO_CMD_FLASH_SET_PARAMS,
            params,
            sizeof(params))
        if nRet != CMD.IS_SUCCESS:
            print("set_flash_params ERROR", nRet)
        else:
            self.get_flash_params()
        return nRet

    # Set current flash mode
    def set_flash_mode(self, mode):
        mode = c_uint(mode)
        nRet = self.uc480.is_IO(
            self.hCam,
            IO_CMD.IS_IO_CMD_FLASH_SET_MODE,
            byref(mode), sizeof(mode))
        if nRet != CMD.IS_SUCCESS:
            print("is_IO Set Flash Mode ERROR", nRet)
        return nRet

    # Get current flash mode
    def get_flash_mode(self, output=False):
        nRet = self.uc480.is_IO(
            self.hCam,
            IO_CMD.IS_IO_CMD_FLASH_GET_MODE,
            byref(self.flash_mode), sizeof(self.flash_mode))
        if nRet != CMD.IS_SUCCESS:
            print("is_IO Get Flash Mode ERROR")
        else:
            print(list(FLASH_MODE.FLASH_MODES.keys())[list(
                FLASH_MODE.FLASH_MODES.values()).index(
                    self.flash_mode.value)])

    def allocate_memory(self, buffers: int = 100):
        if len(self.MemInfo) > 0:
            self.free_memory()

        for x in range(buffers):
            self.MemInfo.append([c_void_p(), c_int(0)])
            ret = self.uc480.is_AllocImageMem(
                self.hCam, c_int(self.width), c_int(self.height),
                self.bit_depth,
                byref(self.MemInfo[x][0]), byref(self.MemInfo[x][1]))
            ret = self.uc480.is_AddToSequence(
                self.hCam, self.MemInfo[x][0], self.MemInfo[x][1])

        ret = self.uc480.is_InitImageQueue(self.hCam, c_int(0))

    def unlock_buffer(self):
        return self.uc480.is_UnlockSeqBuf(
            self.hCam, self.current_id, self.current_buffer)

    # Activates the camera's live video mode (free run mode)
    def start_live_capture(self):
        nRet = self.uc480.is_CaptureVideo(self.hCam, IS_DONT_WAIT)
        if nRet != CMD.IS_SUCCESS:
            print("is_CaptureVideo ERROR")
        else:
            self.capture_video = True
        return nRet

    # Stops the camera's live video mode (free run mode)
    def stop_live_capture(self):
        nRet = self.uc480.is_StopLiveVideo(self.hCam, IS_FORCE_VIDEO_STOP)
        if nRet != CMD.IS_SUCCESS:
            print("is_StopLiveVideo ERROR")
        else:
            self.capture_video = False
        return nRet

    def get_pitch(self):
        x = c_int()
        y = c_int()
        bits = c_int()

        pc_mem = c_void_p()
        pid = c_int()
        self.uc480.is_GetActiveImageMem(self.hCam, byref(pc_mem), byref(pid))
        self.uc480.is_InquireImageMem(
            self.hCam, pc_mem, pid,
            byref(x), byref(y), byref(bits), byref(self.pitch))

        return self.pitch.value

    # Releases an image memory that was allocated using is_AllocImageMem()
    # and removes it from the driver management
    def free_memory(self):
        nRet = 0
        for x in range(len(self.MemInfo)):
            nRet += self.uc480.is_FreeImageMem(
                self.hCam, byref(self.MemInfo[x][0]), self.MemInfo[x][1])
        self.MemInfo.clear()
        self.memory_allocated = False
        return nRet

    def is_WaitForNextImage(self, wait=0, log=True):
        nret = self.uc480.is_WaitForNextImage(
            self.hCam, wait,
            byref(self.current_buffer), byref(self.current_id))
        if nret == CMD.IS_SUCCESS:
            if log:
                logging.debug("is_WaitForNextImage, IS_SUCCESS: {}"
                              .format(nret))
        elif nret == IS_TIMED_OUT:
            if log:
                logging.debug("is_WaitForNextImage, IS_TIMED_OUT: {}"
                              .format(nret))
        elif nret == IS_CAPTURE_STATUS:
            if log:
                logging.debug("is_WaitForNextImage, IS_CAPTURE_STATUS: {}"
                              .format(nret))
            self.CaptureStatusInfo = UC480_CAPTURE_STATUS_INFO()
            nRet = self.uc480.is_CaptureStatus(
                self.hCam, CS_CMD.IS_CAPTURE_STATUS_INFO_CMD_GET,
                self.CaptureStatusInfo, sizeof(self.CaptureStatusInfo))
            if nRet == CMD.IS_SUCCESS:
                self.uc480.is_CaptureStatus(
                    self.hCam, CS_CMD.IS_CAPTURE_STATUS_INFO_CMD_RESET,
                    None, 0)
        return nret

    def get_data(self):
        # In order to display the image in an OpenCV window we need to...
        # ...extract the data of our image memory
        return get_data(
            self.current_buffer,
            self.width, self.height,
            self.bit_depth,
            self.pitch.value, copy=False)

    # Disables the hCam camera handle and releases the data structures
    # and memory areas taken up by the camera
    def dispose(self):
        nRet = self.uc480.is_ExitCamera(self.hCam)
        return nRet


# cam = thorlabs_camera()
# nRet = cam.initialize()

# if nRet == CMD.IS_SUCCESS:
#     cam.set_AOI(0, int((cam.height - 20) / 2), cam.width, 20)

#     cam.set_pixel_clock(cam.pixel_clock_list[-1])
#     cam.get_framerate_range()
#     cam.set_framerate(50)
#     cam.get_exposure_range()
#     cam.set_exposure(3)

#     cam.allocate_memory()
#     cam.start_live_capture()

#     start_chrono = time.time()
#     counter = 0
#     current_buffer = c_void_p()
#     current_id = c_int()
#     m_MissingTrgCounter = 0

#     # while (nRet == ueye.IS_SUCCESS) and (counter < chrono_time):
#     while (True):
#         start_time = time.time()
#         counter = start_time - start_chrono

#         nret = cam.is_WaitForNextImage(cam.hCam, 500)
#         if nret == CMD.IS_SUCCESS:
#             logging.debug(
#                 "is_WaitForNextImage, Status IS_SUCCESS: {}".format(nret))
#             pitch = cam.get_pitch()

#             counter = counter + 1

#             array = get_data(
#                 current_buffer,
#                 cam.width, cam.height,
#                 cam.bit_depth, pitch, copy=False)
#             frame = np.reshape(
#                 array,
#                 (cam.height, cam.width,
#                  cam.bytes_per_pixel))
#             # ...resize the image by a half
#             frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#             # ...and finally display it
#             cv2.imshow("SimpleLive_Python_uEye_OpenCV", frame)
#             # -------------------------------------------------------------
#             # Print FPS here

#             # print('FPS: ' , FPS)
#             # print("FPS: ", int(1.0 / (time.time() - start_time)))

#             cam.uc480.is_UnlockSeqBuf(cam.hCam, current_id, current_buffer)
#            logging.debug(
#                 "is_UnlockSeqBuf, current_id: {}".format(current_id))

#         if nret == IS_TIMED_OUT:
#             logging.debug(
#                 "is_WaitForNextImage, Status IS_TIMED_OUT: {}".format(nret))
#             logging.error("current_buffer: {}".format(current_buffer))
#             logging.error("current_id: {}".format(current_id))

#         if nret == IS_CAPTURE_STATUS:
#             logging.debug(
#                 "is_WaitForNextImage, IS_CAPTURE_STATUS: {}".format(nret))

#             CaptureStatusInfo = UC480_CAPTURE_STATUS_INFO()
#             cam.uc480.is_CaptureStatus(
#                 cam.hCam, CS_CMD.IS_CAPTURE_STATUS_INFO_CMD_GET,
#                 CaptureStatusInfo, sizeof(CaptureStatusInfo))

#             missedTrigger = c_ulong(0)
#             TriggerCnt = c_ulong()  # IS_EXT_TRIGGER_EVENT_CNT
#             missedTrigger = cam.uc480.is_CameraStatus(
#                 cam.hCam, IS_TRIGGER_MISSED, IS_GET_STATUS)
#             m_MissingTrgCounter += missedTrigger

#             logging.error("current_buffer: {}".format(current_buffer))
#             logging.error("current_id: {}".format(current_id))

#         # Press q if you want to end the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cam.stop_live_capture()
#     cam.free_memory()
#     cam.close()
#     # thorlabs_camera.get_camera_list(True)
