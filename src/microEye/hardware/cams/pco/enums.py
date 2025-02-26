from enum import Enum
from typing import Union


class DescriptionKeys(Enum):
    SERIAL = ('serial', int)
    '''Serial number of the camera'''

    TYPE = ('type', str)
    '''Sensor type'''

    SUB_TYPE = ('sub type', int)
    '''Sensor sub type'''

    INTERFACE_TYPE = ('interface type', str)
    '''Interface type'''

    MIN_EXPOSURE_TIME = ('min exposure time', float)
    '''Minimal possible exposure time [s]'''

    MAX_EXPOSURE_TIME = ('max exposure time', float)
    '''Maximal possible exposure time [s]'''

    MIN_EXPOSURE_STEP = ('min exposure step', float)
    '''Minimal possible exposure step [s]'''

    MIN_DELAY_TIME = ('min delay time', float)
    '''Minimal possible delay time [s]'''

    MAX_DELAY_TIME = ('max delay time', float)
    '''Maximal possible delay time [s]'''

    MIN_DELAY_STEP = ('min delay step', float)
    '''Minimal possible delay step [s]'''

    MIN_WIDTH = ('min width', int)
    '''Minimal possible image width (hardware ROI)'''

    MIN_HEIGHT = ('min height', int)
    '''Minimal possible image height (hardware ROI)'''

    MAX_WIDTH = ('max width', int)
    '''Maximal possible image width (hardware ROI)'''

    MAX_HEIGHT = ('max height', int)
    '''Maximal possible image height (hardware ROI)'''

    ROI_STEPS = ('roi steps', tuple[int, int])
    '''Hardware ROI stepping as tuple of (horz, vert)'''

    ROI_IS_HORZ_SYMMETRIC = ('roi is horz symmetric', bool)
    '''Flag if hardware ROI has to be horizontally symmetric'''

    ROI_IS_VERT_SYMMETRIC = ('roi is vert symmetric', bool)
    '''Flag if hardware ROI has to be vertically symmetric'''

    BIT_RESOLUTION = ('bit resolution', int)
    '''Bit-resolution of the sensor'''

    HAS_TIMESTAMP = ('has timestamp', bool)
    '''Flag if camera supports the timestamp setting'''

    HAS_ASCII_ONLY_TIMESTAMP = ('has ascii-only timestamp', bool)
    '''Flag if camera supports setting the timestamp to ascii-only'''

    PIXELRATES = ('pixelrates', list[int])
    '''list containing all possible pixelrate frequencies (index 0 is default)'''

    HAS_ACQUIRE = ('has acquire', bool)
    '''Flag if camera supports the acquire mode setting'''

    HAS_EXTERN_ACQUIRE = ('has extern acquire', bool)
    '''Flag if camera supports the external acquire setting'''

    HAS_METADATA = ('has metadata', bool)
    '''Flag if metadata can be activated for the camera'''

    HAS_RAM = ('has ram', bool)
    '''Flag if camera has internal memory'''

    BINNING_HORZ_VEC = ('binning horz vec', list[int])
    '''list containing all possible horizontal binning values'''

    BINNING_VERT_VEC = ('binning vert vec', list[int])
    '''list containing all possible vertical binning values'''

    HAS_AVERAGE_BINNING = ('has average binning', bool)
    '''Flag if camera supports average binning'''


class ConfigKeys(Enum):
    EXPOSURE_TIME = ('exposure time', float)
    '''Exposure time [s]'''

    DELAY_TIME = ('delay time', float)
    '''Delay time [s]'''

    ROI = ('roi', tuple) # tuple[int, int, int, int])
    '''Hardware ROI as tuple of (x0, y0, x1, y1)'''

    TIMESTAMP = ('timestamp', str)
    '''Timestamp mode'''

    PIXEL_RATE = ('pixel rate', int)
    '''Pixelrate'''

    TRIGGER = ('trigger', str)
    '''Trigger mode'''

    ACQUIRE = ('acquire', str)
    '''Acquire mode'''

    METADATA = ('metadata', str)
    '''Metadata mode'''

    NOISE_FILTER = ('noise filter', str)
    '''Noise filter mode'''

    BINNING = ('binning', tuple) # tuple[int, int, str])
    '''Binning setting as tuple of (horz, vert, mode)'''

    AUTO_EXPOSURE = ('auto exposure', tuple) # tuple[str, int, int])
    '''Auto-Exposure setting as tuple of (region-type, min exposure, max exposure)'''


class RecorderModes(Enum):
    SEQUENCE = ('sequence', 'Memory', True)
    '''Record a sequence of images.'''

    SEQUENCE_NON_BLOCKING = ('sequence non blocking', 'Memory', False)
    '''Record a sequence of images, do not wait until record is finished.'''

    RING_BUFFER = ('ring buffer', 'Memory', False)
    '''Continuously record images in a ringbuffer,
    once the buffer is full, old images are overwritten.'''

    FIFO = ('fifo', 'Memory', False)
    '''Record images in fifo mode, i.e. you will always read images sequentially and
    once the buffer is full, recording will pause until older images have been read.'''

    SEQUENCE_DPCORE = ('sequence dpcore', 'Memory', True)
    '''Same as sequence, but with DotPhoton preparation enabled.'''

    SEQUENCE_NON_BLOCKING_DPCORE = ('sequence non blocking dpcore', 'Memory', False)
    '''Same as sequence_non_blocking, but with DotPhoton preparation enabled.'''

    RING_BUFFER_DPCORE = ('ring buffer dpcore', 'Memory', False)
    '''Same as ring_buffer, but with DotPhoton preparation enabled.'''

    FIFO_DPCORE = ('fifo dpcore', 'Memory', False)
    '''Same as fifo, but with DotPhoton preparation enabled.'''

    TIF = ('tif', 'File', False)
    '''Record images directly as tif files.'''

    MULTITIF = ('multitif', 'File', False)
    '''Record images directly as one or more multitiff file(s).'''

    PCORAW = ('pcoraw', 'File', False)
    '''Record images directly as one pcoraw file.'''

    DICOM = ('dicom', 'File', False)
    '''Record images directly as dicom files.'''

    MULTIDICOM = ('multidicom', 'File', False)
    '''Record images directly as one or more multidicom file(s).'''

    CAMRAM_SEGMENT = ('camram_segment', 'Camera RAM', False)
    '''Record images to camera memory. Stops when segment is full.'''

    CAMRAM_RING = ('camram_ring', 'Camera RAM', False)
    '''Record images to camera memory. Ram segment is used as ring buffer.'''


class ImageFormats(Enum):
    MONO8 = 'mono8'
    '''Get image as 8 bit grayscale data.'''

    MONO16 = 'mono16'
    '''Get image as 16 bit grayscale/raw data.'''

    BGR8 = 'bgr8'
    '''Get image as 24 bit color data in bgr format.'''

    RGB8 = 'rgb8'
    '''Get image as 24 bit color data in rgb format.'''

    BGRA8 = 'bgra8'
    '''Get image as 32 bit color data (with alpha channel) in bgra format.'''

    RGBA8 = 'rgba8'
    '''Get image as 32 bit color data (with alpha channel) in rgba format.'''

    BGR16 = 'bgr16'
    '''Get image as 48 bit color data in bgr format
    (only possible for color cameras).'''

    RGB16 = 'rgb16'
    '''Get image as 48 bit color data in rgb format
    (only possible for color cameras).'''


class ExposureRegionType(Enum):
    BALANCED = 'balanced'
    '''Balanced exposure region'''

    CENTER_BASED = 'center based'
    '''Center based exposure region'''

    CORNER_BASED = 'corner based'
    '''Corner based exposure region'''

    FULL = 'full'
    '''Full exposure region'''

class ConvertControlKeys(Enum):
    SHARPEN = (
        'sharpen',
        bool,
        [ImageFormats.MONO8, ImageFormats.BGR8, ImageFormats.BGR16],
    )
    '''Sharpen: <bool> Supported formats: Mono8, BGR8, BGR16'''

    ADAPTIVE_SHARPEN = (
        'adaptive_sharpen',
        bool,
        [ImageFormats.MONO8, ImageFormats.BGR8, ImageFormats.BGR16],
    )
    '''Adaptive Sharpen: <bool> Supported formats: Mono8, BGR8, BGR16'''

    FLIP_VERTICAL = (
        'flip_vertical',
        bool,
        [ImageFormats.MONO8, ImageFormats.BGR8, ImageFormats.BGR16],
    )
    '''Flip Vertical: <bool> Supported formats: Mono8, BGR8, BGR16'''

    AUTO_MINMAX = (
        'auto_minmax',
        bool,
        [ImageFormats.MONO8, ImageFormats.BGR8, ImageFormats.BGR16],
    )
    '''Auto MinMax: <bool> Supported formats: Mono8, BGR8, BGR16'''

    ADD_CONV_FLAGS = (
        'add_conv_flags',
        int,
        [ImageFormats.MONO8, ImageFormats.BGR8, ImageFormats.BGR16],
    )
    '''Add Conv Flags: <int> Supported formats: Mono8, BGR8, BGR16'''

    MIN_LIMIT = (
        'min_limit',
        int,
        [ImageFormats.MONO8, ImageFormats.BGR8, ImageFormats.BGR16],
    )
    '''Min Limit: <int> Supported formats: Mono8, BGR8, BGR16'''

    MAX_LIMIT = (
        'max_limit',
        int,
        [ImageFormats.MONO8, ImageFormats.BGR8, ImageFormats.BGR16],
    )
    '''Max Limit: <int> Supported formats: Mono8, BGR8, BGR16'''

    GAMMA = (
        'gamma',
        float,
        [ImageFormats.MONO8, ImageFormats.BGR8, ImageFormats.BGR16],
    )
    '''Gamma: <double> Supported formats: Mono8, BGR8, BGR16'''

    CONTRAST = (
        'contrast',
        int,
        [ImageFormats.MONO8, ImageFormats.BGR8, ImageFormats.BGR16],
    )
    '''Contrast: <int> Supported formats: Mono8, BGR8, BGR16'''

    COLOR_TEMPERATURE = (
        'color_temperature',
        int,
        [ImageFormats.BGR8, ImageFormats.BGR16],
    )
    '''Color Temperature: <int> Supported formats: BGR8, BGR16'''

    COLOR_SATURATION = (
        'color_saturation',
        int,
        [ImageFormats.BGR8, ImageFormats.BGR16],
    )
    '''Color Saturation: <int> Supported formats: BGR8, BGR16'''

    COLOR_VIBRANCE = ('color_vibrance', int, [ImageFormats.BGR8, ImageFormats.BGR16])
    '''Color Vibrance: <int> Supported formats: BGR8, BGR16'''

    COLOR_TINT = ('color_tint', int, [ImageFormats.BGR8, ImageFormats.BGR16])
    '''Color Tint: <int> Supported formats: BGR8, BGR16'''

    LUT_FILE = ('lut_file', str, [ImageFormats.BGR8])
    '''LUT File: <file_path> Supported formats: BGR8 for non-colored cameras'''

class MetadataKeys(Enum):
    DATA_FORMAT = ('data format', str)
    '''Data format: <str> Supported formats:
    "Mono8", "Mono16", "BGR8", "BGR16", "CompressedMono8" '''

    RECORDER_IMAGE_NUMBER = ('recorder image number', int)
    '''Recorder image number: <int> from pco.recorder'''

    TIMESTAMP = ('timestamp', dict[str, Union[int, float]])
    '''Timestamp: <dict> {
    "image counter": <int>, "year": <int>, "month": <int>,"day": <int>,
    "hour": <int>, "minute": <int>, "second": <float>, "status": <int>}'''

    VERSION = ('version', int)
    '''Version: <int> from PCO_METADATA_STRUCT'''

    EXPOSURE_TIME = ('exposure time', int)
    '''Exposure time: <int> from PCO_METADATA_STRUCT'''

    FRAMERATE = ('framerate', float)
    '''Framerate: <float> in Hz'''

    SENSOR_TEMPERATURE = ('sensor temperature', int)
    '''Sensor temperature: <int> from PCO_METADATA_STRUCT'''

    PIXEL_CLOCK = ('pixel clock', int)
    '''Pixel clock: <int> from PCO_METADATA_STRUCT'''

    CONVERSION_FACTOR = ('conversion factor', int)
    '''Conversion factor: <int> from PCO_METADATA_STRUCT'''

    SERIAL_NUMBER = ('serial number', int)
    '''Serial number: <int> from PCO_METADATA_STRUCT'''

    CAMERA_TYPE = ('camera type', int)
    '''Camera type: <int> from PCO_METADATA_STRUCT'''

    BIT_RESOLUTION = ('bit resolution', int)
    '''Bit resolution: <int> from PCO_METADATA_STRUCT'''

    SYNC_STATUS = ('sync status', int)
    '''Sync status: <int> from PCO_METADATA_STRUCT'''

    DARK_OFFSET = ('dark offset', int)
    '''Dark offset: <int> from PCO_METADATA_STRUCT'''

    TRIGGER_MODE = ('trigger mode', int)
    '''Trigger mode: <int> from PCO_METADATA_STRUCT'''

    DOUBLE_IMAGE_MODE = ('double image mode', int)
    '''Double image mode: <int> from PCO_METADATA_STRUCT'''

    CAMERA_SYNC_MODE = ('camera sync mode', int)
    '''Camera sync mode: <int> from PCO_METADATA_STRUCT'''

    IMAGE_TYPE = ('image type', int)
    '''Image type: <int> from PCO_METADATA_STRUCT'''

    COLOR_PATTERN = ('color pattern', int)
    '''Color pattern: <int> from PCO_METADATA_STRUCT'''

    IMAGE_SIZE = ('image size', int)
    '''Image size: <int> from PCO_METADATA_STRUCT'''

    BINNING = ('binning', int)
    '''Binning: <int> from PCO_METADATA_STRUCT'''

    CAMERA_SUBTYPE = ('camera subtype', int)
    '''Camera subtype: <int> from PCO_METADATA_STRUCT'''

    EVENT_NUMBER = ('event number', int)
    '''Event number: <int> from PCO_METADATA_STRUCT'''

    IMAGE_SIZE_OFFSET = ('image size offset', int)
    '''Image size offset: <int> from PCO_METADATA_STRUCT'''

    TIMESTAMP_BCD = ('timestamp bcd', dict[str, Union[int, float]])
    '''Timestamp BCD: <dict> {
    "image counter": <int>, "year": <int>, "month": <int>, "day": <int>,
    "hour": <int>, "minute": <int>, "second": <float>, "status": <int>}'''
