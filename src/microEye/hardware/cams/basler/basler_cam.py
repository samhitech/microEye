import contextlib
import traceback
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
from pypylon import genicam, pylon

from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.micam import miCamera

ICOMMANDS = [
    # 'AcquisitionAbort',
    # 'AcquisitionStart',
    # 'AcquisitionStop',
    # 'BslCenterX',
    # 'BslCenterY',
    # 'BslErrorReportNext',
    # 'BslLightControlEnumerateDevices',
    # 'BslLightControlOvertriggerCountReset',
    # 'BslLightDeviceErrorStatusReadAndClear',
    # 'BslLightDeviceNewIDSave',
    # 'BslSensorOff',
    # 'BslSensorOn',
    # 'BslSensorStandby',
    # 'BslSerialReceive',
    # 'BslSerialRxBreakReset',
    # 'BslSerialTransmit',
    # 'BslStaticDefectPixelCorrectionReload',
    # 'CounterReset',
    # 'DeviceFeaturePersistenceEnd',
    # 'DeviceFeaturePersistenceStart',
    # 'DeviceRegistersStreamingEnd',
    # 'DeviceRegistersStreamingStart',
    # 'DeviceReset',
    # 'FileOperationExecute',
    # 'SequencerSetLoad',
    # 'SequencerSetSave',
    'SoftwareSignalPulse',
    # 'TestEventGenerate',
    'TimerReset',
    # 'TimestampLatch',
    # 'TriggerEventTest',
    'TriggerSoftware',
    # 'UserSetLoad',
    # 'UserSetSave',
]

IINTEGERS = [
    # 'AcquisitionBurstFrameCount',
    # 'AutoFunctionROIHeight',
    # 'AutoFunctionROIOffsetX',
    # 'AutoFunctionROIOffsetY',
    # 'AutoFunctionROIWidth',
    'BinningHorizontal',
    'BinningVertical',
    # 'BslChunkTimestampValue',
    # 'BslDeviceLinkCurrentThroughput',
    # 'BslErrorReportValue',
    # 'BslImageCompressionBCBCompressedPayloadSize',
    # 'BslImageCompressionBCBDecompressedImageSize',
    # 'BslImageCompressionBCBExtraChunkDataSize',
    # 'BslImageCompressionBCBMode',
    # 'BslImageCompressionBCBVersion',
    # 'BslImageCompressionLastSize',
    # 'BslLightControlOvertriggerCount',
    # 'BslLightDeviceErrorCode',
    # 'BslLineOverloadStatusAll',
    # 'BslMultipleROIColumnOffset',
    # 'BslMultipleROIColumnSize',
    # 'BslMultipleROIRowOffset',
    # 'BslMultipleROIRowSize',
    # 'BslSerialTransferLength',
    # 'BslStaticDefectPixelCorrectionMaxDefects',
    # 'BslTemperatureStatusErrorCount',
    # 'ChunkCounterValue',
    # 'ChunkFrameID',
    # 'ChunkLineStatusAll',
    # 'ChunkPayloadCRC16',
    # 'ChunkSequencerSetActive',
    # 'ChunkTimestamp',
    # 'CounterDuration',
    # 'CounterValue',
    # 'DeviceGenCPVersionMajor',
    # 'DeviceGenCPVersionMinor',
    'DeviceLinkSpeed',
    'DeviceLinkThroughputLimit',
    # 'DeviceManifestSchemaMajorVersion',
    # 'DeviceManifestSchemaMinorVersion',
    # 'DeviceManifestXMLMajorVersion',
    # 'DeviceManifestXMLMinorVersion',
    # 'DeviceManifestXMLSubMinorVersion',
    # 'DeviceSFNCVersionMajor',
    # 'DeviceSFNCVersionMinor',
    # 'DeviceSFNCVersionSubMinor',
    # 'DeviceTLVersionMajor',
    # 'DeviceTLVersionMinor',
    # 'DeviceTLVersionSubMinor',
    # 'DigitalShift',
    # 'EventExposureEnd',
    # 'EventExposureEndFrameID',
    # 'EventExposureEndTimestamp',
    # 'EventFrameBufferOverrun',
    # 'EventFrameBufferOverrunTimestamp',
    # 'EventFrameStart',
    # 'EventFrameStartFrameID',
    # 'EventFrameStartTimestamp',
    # 'EventFrameTriggerMissed',
    # 'EventFrameTriggerMissedTimestamp',
    # 'EventOverrun',
    # 'EventOverrunTimestamp',
    # 'EventTemperatureStatusChanged',
    # 'EventTemperatureStatusChangedTimestamp',
    # 'EventTest',
    # 'EventTestTimestamp',
    # 'FileAccessLength',
    # 'FileAccessOffset',
    # 'FileOperationResult',
    # 'GrabLoopThreadPriority',
    # 'GrabLoopThreadTimeout',
    # 'Height',
    # 'HeightMax',
    # 'InternalGrabEngineThreadPriority',
    # 'LUTIndex',
    # 'LUTValue',
    # 'LineStatusAll',
    # 'MaxNumBuffer',
    # 'MaxNumGrabResults',
    # 'MaxNumQueuedBuffer',
    # 'NumEmptyBuffers',
    # 'NumQueuedBuffers',
    # 'NumReadyBuffers',
    # 'OffsetX',
    # 'OffsetY',
    # 'OutputQueueSize',
    # 'PayloadSize',
    # 'PixelDynamicRangeMax',
    # 'PixelDynamicRangeMin',
    # 'SIPayloadFinalTransfer1Size',
    # 'SIPayloadFinalTransfer2Size',
    # 'SIPayloadTransferCount',
    # 'SIPayloadTransferSize',
    # 'SensorHeight',
    # 'SensorWidth',
    # 'SequencerPathSelector',
    # 'SequencerSetActive',
    # 'SequencerSetNext',
    # 'SequencerSetSelector',
    # 'SequencerSetStart',
    # 'StaticChunkNodeMapPoolSize',
    # 'TLParamsLocked',
    # 'TestPendingAck',
    # 'TimestampLatchValue',
    # 'UserDefinedValue',
    # 'UserOutputValueAll',
    # 'Width',
    # 'WidthMax',
]

ISTRINGS = [
    'BslLightDeviceFirmwareVersion',
    'BslLightDeviceModelName',
    'DeviceFamilyName',
    'DeviceFirmwareVersion',
    'DeviceManufacturerInfo',
    'DeviceModelName',
    'DeviceSerialNumber',
    'DeviceUserID',
    'DeviceVendorName',
    'DeviceVersion',
]

IBOOLEANS = [
    'AcquisitionFrameRateEnable',
    # 'AcquisitionStatus',
    # 'AutoFunctionROIHighlight',
    # 'AutoFunctionROIUseBrightness',
    # 'BslErrorPresent',
    # 'BslLineOverloadStatus',
    # 'BslMultipleROIColumnsEnable',
    # 'BslMultipleROIRowsEnable',
    # 'BslSerialRxBreak',
    # 'BslSerialRxFifoOverflow',
    # 'BslSerialRxParityError',
    # 'BslSerialRxStopBitError',
    # 'BslSerialTxBreak',
    # 'BslSerialTxFifoEmpty',
    # 'BslSerialTxFifoOverflow',
    # 'ChunkEnable',
    # 'ChunkModeActive',
    # 'ChunkNodeMapsEnable',
    # 'GrabCameraEvents',
    # 'GrabLoopThreadPriorityOverride',
    # 'GrabLoopThreadUseTimeout',
    # 'InternalGrabEngineThreadPriorityOverride',
    # 'LUTEnable',
    'LineInverter',
    'LineStatus',
    # 'MonitorModeActive',
    'ReverseX',
    'ReverseY',
    # 'UserOutputValue',
]

IFLOATS = [
    'AcquisitionFrameRate',
    'AutoExposureTimeLowerLimit',
    'AutoExposureTimeUpperLimit',
    'AutoGainLowerLimit',
    'AutoGainUpperLimit',
    # 'AutoTargetBrightness',
    'BlackLevel',
    'BslBrightness',
    'BslContrast',
    'BslEffectiveExposureTime',
    'BslExposureStartDelay',
    'BslFlashWindowDelay',
    'BslFlashWindowDuration',
    # 'BslImageCompressionLastRatio',
    # 'BslImageCompressionRatio',
    # 'BslInputFilterTime',
    # 'BslInputHoldOffTime',
    # 'BslLightDeviceBrightness',
    # 'BslLightDeviceCurrent',
    # 'BslLightDeviceDutyCycle',
    # 'BslLightDeviceOverdriveLimit',
    # 'BslLightDeviceStrobeDuration',
    # 'BslNoiseReduction',
    # 'BslResultingAcquisitionFrameRate',
    # 'BslResultingFrameBurstRate',
    # 'BslResultingTransferFrameRate',
    # 'BslScalingFactor',
    # 'BslSharpnessEnhancement',
    # 'BslTemperatureMax',
    # 'ChunkExposureTime',
    # 'ChunkGain',
    # 'DeviceTemperature',
    # 'ExposureTime',
    'Gain',
    'Gamma',
    'ResultingFrameRate',
    'SensorReadoutTime',
    'TimerDelay',
    'TimerDuration',
    'TimerTriggerArmDelay',
    'TriggerDelay',
]

IENUMS = [
    'AcquisitionMode',
    'AcquisitionStatusSelector',
    'AcquisitionStopMode',
    # 'AutoFunctionProfile',
    # 'AutoFunctionROISelector',
    'BinningHorizontalMode',
    'BinningSelector',
    'BinningVerticalMode',
    'BlackLevelSelector',
    # 'BslAcquisitionBurstMode',
    # 'BslAcquisitionStopMode',
    # 'BslChunkAutoBrightnessStatus',
    # 'BslChunkTimestampSelector',
    # 'BslContrastMode',
    'BslConversionGainMode',
    'BslDeviceFirmwareType',
    'BslExposureTimeMode',
    # 'BslLightControlErrorSummary',
    # 'BslLightControlMode',
    # 'BslLightControlTriggerActivation',
    # 'BslLightControlTriggerSource',
    # 'BslLightDeviceControlMode',
    # 'BslLightDeviceErrorStatus',
    # 'BslLightDeviceNewID',
    # 'BslLightDeviceOperationMode',
    # 'BslLightDeviceSelector',
    # 'BslLightDeviceStrobeMode',
    # 'BslLineConnection',
    # 'BslMultipleROIColumnSelector',
    # 'BslMultipleROIRowSelector',
    # 'BslSensorBitDepth',
    # 'BslSensorBitDepthMode',
    # 'BslSensorState',
    # 'BslSerialBaudRate',
    # 'BslSerialNumberOfDataBits',
    # 'BslSerialNumberOfStopBits',
    # 'BslSerialParity',
    # 'BslSerialRxSource',
    # 'BslSerialTransmitMode',
    # 'BslStaticDefectPixelCorrectionFileStatus',
    'BslStaticDefectPixelCorrectionMode',
    # 'BslTemperatureStatus',
    # 'BslTransferBitDepth',
    # 'BslTransferBitDepthMode',
    # 'BslUSBPowerSource',
    # 'BslUSBSpeedMode',
    # 'ChunkCounterSelector',
    # 'ChunkExposureTimeSelector',
    # 'ChunkGainSelector',
    # 'ChunkSelector',
    # 'CounterEventActivation',
    # 'CounterEventSource',
    # 'CounterResetActivation',
    # 'CounterResetSource',
    # 'CounterSelector',
    # 'CounterStatus',
    # 'CounterTriggerActivation',
    # 'CounterTriggerSource',
    # 'DeviceCharacterSet',
    'DeviceIndicatorMode',
    'DeviceLinkThroughputLimitMode',
    # 'DeviceRegistersEndianness',
    # 'DeviceScanType',
    # 'DeviceTLType',
    'DeviceTemperatureSelector',
    # 'EventNotification',
    # 'EventSelector',
    # 'EventTemperatureStatusChangedStatus',
    'ExposureAuto',
    'ExposureMode',
    'ExposureTimeMode',
    'ExposureTimeSelector',
    # 'FileOpenMode',
    # 'FileOperationSelector',
    # 'FileOperationStatus',
    # 'FileSelector',
    'GainAuto',
    'GainSelector',
    'ImageCompressionMode',
    'ImageCompressionRateOption',
    'LUTSelector',
    'LineFormat',
    'LineMode',
    'LineSelector',
    'LineSource',
    'PixelFormat',
    'PixelSize',
    'SensorShutterMode',
    # 'SequencerConfigurationMode',
    # 'SequencerMode',
    # 'SequencerTriggerActivation',
    # 'SequencerTriggerSource',
    # 'ServiceBoardIdSelector',
    'SoftwareSignalSelector',
    # 'TestPattern',
    'TimerSelector',
    'TimerStatus',
    'TimerTriggerActivation',
    'TimerTriggerSource',
    'TriggerActivation',
    'TriggerMode',
    'TriggerSelector',
    'TriggerSource',
    #     'UserDefinedValueSelector',
    #     'UserOutputSelector',
    #     'UserSetDefault',
    #     'UserSetSelector',
]


class BaslerParams(Enum):
    FREERUN = 'Acquisition.Freerun'
    STOP = 'Acquisition.Stop'
    # Acquisition
    TRIGGER_SOFTWARE = 'Acquisition.TriggerSoftware'

    # Acquisition Settings
    ACQUISITION_FRAMERATE = 'Acquisition Settings.AcquisitionFrameRate'
    ACQUISITION_FRAMERATE_ENABLE = 'Acquisition Settings.AcquisitionFrameRateEnable'
    ACQUISITION_MODE = 'Acquisition Settings.AcquisitionMode'

    BINNING_SELECTOR = 'Acquisition Settings.BinningSelector'
    BINNING_HORIZONTAL = 'Acquisition Settings.BinningHorizontal'
    BINNING_VERTICAL = 'Acquisition Settings.BinningVertical'
    BINNING_HORIZONTAL_MODE = 'Acquisition Settings.BinningHorizontalMode'
    BINNING_VERTICAL_MODE = 'Acquisition Settings.BinningVerticalMode'

    BLACK_LEVEL = 'Acquisition Settings.BlackLevel'
    CONVERSION_GAIN_MODE = 'Acquisition Settings.BslConversionGainMode'
    DEFECT_PIXEL_CORRECTION_MODE = (
        'Acquisition Settings.BslStaticDefectPixelCorrectionMode'
    )
    EXPOSURE_AUTO = 'Acquisition Settings.ExposureAuto'
    EXPOSURE_MODE = 'Acquisition Settings.ExposureMode'
    GAIN = 'Acquisition Settings.Gain'
    GAIN_AUTO = 'Acquisition Settings.GainAuto'
    GAIN_SELECTOR = 'Acquisition Settings.GainSelector'
    GAMMA = 'Acquisition Settings.Gamma'
    PIXEL_FORMAT = 'Acquisition Settings.PixelFormat'
    PIXEL_SIZE = 'Acquisition Settings.PixelSize'
    RESULTING_FRAMERATE = 'Acquisition Settings.ResultingFrameRate'
    REVERSE_X = 'Acquisition Settings.ReverseX'
    REVERSE_Y = 'Acquisition Settings.ReverseY'
    SENSOR_READOUT_TIME = 'Acquisition Settings.SensorReadoutTime'
    SENSOR_SHUTTER_MODE = 'Acquisition Settings.SensorShutterMode'
    TRIGGER_ACTIVATION = 'Acquisition Settings.TriggerActivation'
    TRIGGER_DELAY = 'Acquisition Settings.TriggerDelay'
    TRIGGER_MODE = 'Acquisition Settings.TriggerMode'
    TRIGGER_SELECTOR = 'Acquisition Settings.TriggerSelector'
    TRIGGER_SOURCE = 'Acquisition Settings.TriggerSource'

    # GPIOs
    LINE_INVERTER = 'GPIOs.LineInverter'
    LINE_MODE = 'GPIOs.LineMode'
    LINE_SELECTOR = 'GPIOs.LineSelector'
    LINE_SOURCE = 'GPIOs.LineSource'
    LINE_STATUS = 'GPIOs.LineStatus'
    LINE_FORMAT = 'GPIOs.LineFormat'
    LINE_CONNECTION = 'GPIOs.BslLineConnection'

    # Stats (Device Info & Strings)
    BSL_LIGHT_DEVICE_FIRMWARE_VERSION = 'Stats.BslLightDeviceFirmwareVersion'
    BSL_LIGHT_DEVICE_MODEL_NAME = 'Stats.BslLightDeviceModelName'
    DEVICE_FAMILY_NAME = 'Stats.DeviceFamilyName'
    DEVICE_FIRMWARE_VERSION = 'Stats.DeviceFirmwareVersion'
    DEVICE_LINK_SPEED = 'Stats.DeviceLinkSpeed'
    DEVICE_LINK_THROUGHPUT_LIMIT = 'Stats.DeviceLinkThroughputLimit'
    DEVICE_MANUFACTURER_INFO = 'Stats.DeviceManufacturerInfo'
    DEVICE_MODEL_NAME = 'Stats.DeviceModelName'
    DEVICE_SERIAL_NUMBER = 'Stats.DeviceSerialNumber'
    DEVICE_USER_ID = 'Stats.DeviceUserID'
    DEVICE_VENDOR_NAME = 'Stats.DeviceVendorName'
    DEVICE_VERSION = 'Stats.DeviceVersion'

    # Timers
    TIMER_DELAY = 'Timers.TimerDelay'
    TIMER_DURATION = 'Timers.TimerDuration'
    TIMER_SELECTOR = 'Timers.TimerSelector'
    TIMER_STATUS = 'Timers.TimerStatus'
    TIMER_TRIGGER_ARM_DELAY = 'Timers.TimerTriggerArmDelay'
    TIMER_TRIGGER_ACTIVATION = 'Timers.TimerTriggerActivation'
    TIMER_TRIGGER_SOURCE = 'Timers.TimerTriggerSource'
    TIMER_RESET = 'Timers.TimerReset'
    SOFTWARE_SIGNAL_SELECTOR = 'Timers.SoftwareSignalSelector'
    SOFTWARE_SIGNAL = 'Timers.SoftwareSignalPulse'

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


class basler_cam(miCamera):
    def __init__(self, FullName=None, **kwargs) -> None:
        super().__init__(FullName)

        self.bytes_per_pixel = 2

        self.cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
            if FullName is None
            else pylon.TlFactory.GetInstance().CreateDevice(FullName)
        )
        self.cam.Open()

        self.initialize()

    def initialize(self):
        '''Initialize the camera'''
        self.cam.UserSetSelector.Value = 'Default'
        self.cam.UserSetLoad.Execute()
        self.cam.BslStaticDefectPixelCorrectionMode.Value = 'Off'
        self.getExposure()

        with contextlib.suppress(Exception):
            self.cam.PixelFormat.Value = 'Mono12'
            self.cam.BlackLevel.Value = 10

        self.name = self.cam.DeviceModelName()
        self.serial = self.cam.DeviceSerialNumber()
        self.interface = self.cam.DeviceInfo.GetTLType()

        self.print_status()

    @classmethod
    def get_camera_list(cls):
        '''
        Gets the list of available Basler cameras

        Returns
        -------
        dict[str, object]
            dictionary containing
            {Camera ID, Device ID, Sensor ID, Status, InUse, Model, Serial}
        '''
        cam_list = []
        instance: pylon.TlFactory = pylon.TlFactory.GetInstance()
        devices = instance.EnumerateDevices()
        if devices:
            for index, device in enumerate(devices):
                cam_list.append(
                    {
                        'Camera ID': device.GetSerialNumber(),
                        'Device ID': index,
                        'Sensor ID': 'NA',
                        'Status': 'NA',
                        'InUse': not instance.IsDeviceAccessible(device),
                        'Model': device.GetModelName(),
                        'Serial': device.GetSerialNumber(),
                        'Driver': device.GetVendorName(),
                        'FullName': device.GetFullName(),
                    }
                )
        return cam_list

    def property_tree(self) -> list[dict[str, Any]]:
        try:
            tree = []

            tree.append(
                {
                    'name': str(BaslerParams.FREERUN),
                    'type': 'action',
                    'parent': CamParams.ACQUISITION,
                }
            )
            tree.append(
                {
                    'name': str(BaslerParams.STOP),
                    'type': 'action',
                    'parent': CamParams.ACQUISITION,
                }
            )

            # iterate through BaslerParams enums
            for prop in BaslerParams:
                name = prop.value.split('.')[-1]

                try:
                    parent = CamParams(prop.value.split('.')[0])
                except ValueError:
                    continue

                if name not in dir(self.cam):
                    continue

                attr = getattr(self.cam, name, None)
                # if attribute is IValue subcalss
                if not isinstance(attr, genicam.IValue):
                    continue

                signal = None

                # get property type based on the attribute being IFloat, IEnumeration...
                if isinstance(attr, genicam.IFloat):
                    python_type = 'float'
                    prop_unit = attr.Unit
                    prop_limits = [attr.Min, attr.Max]
                elif isinstance(attr, genicam.IEnumeration):
                    python_type = 'list'
                    prop_unit = None
                    prop_limits = attr.Symbolics
                elif isinstance(attr, genicam.IInteger):
                    python_type = 'int'
                    prop_unit = attr.Unit
                    prop_limits = [attr.Min, attr.Max]
                elif isinstance(attr, genicam.IBoolean):
                    python_type = 'bool'
                    prop_unit = None
                    prop_limits = None
                elif isinstance(attr, genicam.IString):
                    python_type = 'str'
                    prop_unit = None
                    prop_limits = None
                elif isinstance(attr, genicam.ICommand):
                    python_type = 'action'
                    prop_unit = None
                    prop_limits = None

                    def signal(attr=attr):
                        return attr.Execute()
                else:
                    continue

                access = attr.GetAccessMode()
                value_flag = access == genicam.RW or access == genicam.RO

                tree.append(
                    {
                        'name': name,
                        'type': python_type,
                        'value': attr.Value if value_flag else None,
                        'enabled': attr.GetAccessMode() == genicam.RW
                        if python_type != 'action'
                        else True,
                        'limits': prop_limits,
                        'suffix': prop_unit,
                        'parent': parent,
                        'signal': signal,
                    }
                )
        except Exception as e:
            tree = []

        return tree

    def update_cam(self, param, path, param_value):
        if path is None:
            return

        param_name = param.name()

        if param_name not in dir(self.cam):
            return

        attr = getattr(self.cam, param_name, None)

        if (
            isinstance(attr, genicam.IValue)
            and attr.GetAccessMode() == genicam.RW
            and isinstance(param_value, type(attr.Value))
        ):
            try:
                attr.SetValue(param_value)
            except Exception as e:
                print(
                    f'Failed to set {param_name} to {param_value}({type(param_value)})'
                )

    @property
    def exposure_increment(self) -> float:
        '''Get the exposure increment of the camera

        Returns
        -------
        float
            The exposure increment of the camera
        '''
        return self.cam.ExposureTime.Inc

    @property
    def exposure_range(self) -> tuple[float, float]:
        '''
        Get the minimum and maximum exposure time of the camera

        Returns
        -------
        tuple[float, float]
            The minimum and maximum exposure time of the camera
        '''
        return (
            self.cam.ExposureTime.Min,
            self.cam.ExposureTime.Max,
        )

    @exposure_range.setter
    def exposure_range(self, value: tuple[float, float]):
        pass

    @property
    def framerate(self) -> float:
        '''Get the current framerate of the camera

        Returns
        -------
        float
            The current framerate of the camera
        '''
        return self.cam.AcquisitionFrameRate()

    @framerate.setter
    def framerate(self, value: float):
        '''Set the current framerate of the camera

        Parameters
        ----------
        value : float
            The desired framerate of the camera
        '''
        self.cam.AcquisitionFrameRate.Value = value

    @property
    def min_framerate(self) -> float:
        '''Get the minimum framerate of the camera

        Returns
        -------
        float
            The minimum framerate of the camera
        '''
        return self.cam.AcquisitionFrameRate.Min

    @property
    def max_framerate(self) -> float:
        '''Get the maximum framerate of the camera

        Returns
        -------
        float
            The maximum framerate of the camera
        '''
        return self.cam.AcquisitionFrameRate.Max

    @property
    def min_dim(self) -> tuple[int, int]:
        '''Get the minimum dimensions of the camera

        Returns
        -------
        tuple[int, int]
            The minimum width and height of the camera
        '''
        return (
            self.cam.Width.Min,
            self.cam.Height.Min,
        )

    @property
    def max_dim(self) -> tuple[int, int]:
        '''Get the maximum dimensions of the camera

        Returns
        -------
        tuple[int, int]
            The maximum width and height of the camera
        '''
        return (
            self.cam.Width.Max,
            self.cam.Height.Max,
        )

    @property
    def height(self):
        '''Get the height of the camera'''
        return self.cam.Height()

    @height.setter
    def height(self, value):
        '''Set the height of the camera'''
        self.cam.Height.Value = value

    @property
    def width(self):
        '''Get the width of the camera ROI'''
        return self.cam.Width()

    @width.setter
    def width(self, value):
        '''Set the width of the camera'''
        self.cam.Width.Value = value

    @property
    def roi_steps(self) -> tuple[int, int]:
        '''Get the steps for the ROI

        Returns
        -------
        tuple[int, int]
            The steps for the ROI in the horizontal and vertical direction
        '''
        return (
            self.cam.Width.Inc,
            self.cam.Height.Inc,
        )

    @property
    def ROI(self) -> tuple[int, int, int, int]:
        '''
        Get the current region of interest (ROI) of the camera

        Returns
        -------
        tuple[int, int, int, int]
            The current region of interest (x, y, width, height),
            Note that x and y are 1-indexed.
        '''
        return (
            self.cam.OffsetX.Value,
            self.cam.OffsetY.Value,
            self.cam.Width.Value,
            self.cam.Height.Value,
        )

    def close(self):
        '''Close the camera'''
        if self.cam is not None:
            self.cam.Close()

    def populate_status(self):
        self.status['Camera'] = {
            'Name': self.cam.DeviceModelName(),
            'Serial': self.cam.DeviceSerialNumber(),
            'Interface': self.cam.DeviceInfo.GetTLType(),
        }

        self.status['Exposure'] = {
            'Value': self.cam.ExposureTime() / 1000,
            'Unit': 'ms',
        }

        self.status['Temperature'] = {
            'Value': self.get_temperature(),
            'Unit': 'Celsius',
        }

    def get_temperature(self) -> float:
        '''Reads out the sensor temperature value

        Returns
        -------
        float
            camera sensor temperature in C.
        '''
        self.temperature = -127
        if self.cam is not None:
            self.temperature = self.cam.DeviceTemperature()
        return self.temperature

    def getExposure(self) -> float:
        '''Get the current exposure time of the camera

        Returns
        -------
        float
            The current exposure time of the camera
        '''
        exp = -127
        try:
            if self.cam is not None:
                self.exposure_current = self.cam.ExposureTime()
                self.exposure_unit = self.cam.ExposureTime.Unit
                return self.exposure_current
        except Exception:
            print('Exposure Get ERROR')
        return exp

    def setExposure(self, exp: float) -> float:
        '''Set the exposure time of the camera

        Parameters
        ----------
        exp : float
            The exposure time to set in seconds

        Returns
        -------
        float
            The current exposure time of the camera
        '''
        try:
            if self.cam is not None:
                self.cam.ExposureTime.Value = exp
                self.exposure_current = self.cam.ExposureTime()
                return self.exposure_current
        except Exception:
            print('Exposure Set ERROR')
        return -127

    def AcquisitionStatus(self):
        '''Get the acquisition status of the camera

        Returns
        -------
        bool
            The acquisition status of the camera
        '''
        return self.cam.AcquisitionStatus()

    @property
    def recorded_image_count(self):
        '''Get the number of recorded images'''
        return self.cam.recorded_image_count

    def get_roi(self):
        return self.ROI

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        '''
        Set the region of interest (ROI) of the camera

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
        '''
        # Align values to increments
        x_inc = self.cam.OffsetX.Inc
        y_inc = self.cam.OffsetY.Inc
        w_inc = self.cam.Width.Inc
        h_inc = self.cam.Height.Inc

        x_aligned = ((x - self.cam.OffsetX.Min) // x_inc) * x_inc + self.cam.OffsetX.Min
        y_aligned = ((y - self.cam.OffsetY.Min) // y_inc) * y_inc + self.cam.OffsetY.Min
        w_aligned = ((width - self.cam.Width.Min) // w_inc) * w_inc + self.cam.Width.Min
        h_aligned = (
            (height - self.cam.Height.Min) // h_inc
        ) * h_inc + self.cam.Height.Min

        self.cam.OffsetX.Value = x_aligned
        self.cam.OffsetY.Value = y_aligned
        self.cam.Width.Value = w_aligned
        self.cam.Height.Value = h_aligned

    def reset_roi(self) -> None:
        '''Reset the region of interest (ROI) of the camera'''
        self.cam.OffsetX.Value = 0
        self.cam.OffsetY.Value = 0
        self.cam.Width.Value = self.cam.Width.Max
        self.cam.Height.Value = self.cam.Height.Max

    def StopGrabbing(self):
        '''Stop the camera'''
        self.cam.StopGrabbing()

    def StartGrabbing(self, strategy: int = 0, grabLoopType: int = 0):
        '''Start the camera

        Parameters
        ----------
        strategy : int
            The grab strategy. See Pylon::InstantCamera::EStrategy for more information.

            - `0`: pylon.GrabStrategy_OneByOne: Retrieve images one at a time, in order.
            - `1`: pylon.GrabStrategy_LatestImageOnly: Only the latest image is kept;
            older images are discarded.
            - `2`: pylon.GrabStrategy_LatestImages: Keep a set number of the
            latest images.
            - `3`: pylon.GrabStrategy_UpcomingImage: Retrieve the next image that will
            be grabbed.

        grabLoopType : int

            - `0`: GrabLoop_ProvidedByInstantCamera
              An additional grab loop thread is used to run the grab loop.
            - `1`: GrabLoop_ProvidedByUser
              The user is responsible for running the grab loop.
        '''
        self.cam.StartGrabbing(strategy, grabLoopType)

    def StartGrabbingMax(
        self, maxImages: int, strategy: int = 0, grabLoopType: int = 0
    ):
        '''Start the camera

        Parameters
        ----------
        maxImages : int
            The count of images to grab. This value must be larger than zero.

        strategy : int
            The grab strategy. See Pylon::InstantCamera::EStrategy for more information.

            - `0`: pylon.GrabStrategy_OneByOne: Retrieve images one at a time, in order.
            - `1`: pylon.GrabStrategy_LatestImageOnly: Only the latest image is kept;
            older images are discarded.
            - `2`: pylon.GrabStrategy_LatestImages: Keep a set number of the
            latest images.
            - `3`: pylon.GrabStrategy_UpcomingImage: Retrieve the next image that will
            be grabbed.

        grabLoopType : int

            - `0`: GrabLoop_ProvidedByInstantCamera
              An additional grab loop thread is used to run the grab loop.
            - `1`: GrabLoop_ProvidedByUser
              The user is responsible for running the grab loop.
        '''
        self.cam.StartGrabbingMax(maxImages, strategy, grabLoopType)

    def IsGrabbing(self) -> bool:
        '''Check if the camera is grabbing

        Returns
        -------
        bool
            True if the camera is grabbing, False otherwise
        '''
        return self.cam.IsGrabbing()

    def GrabOne(self) -> Optional[np.ndarray]:
        '''Grab one image from the camera

        Returns
        -------
        Optional[np.ndarray]
            The result of the grabbing or None if failed
        '''
        if self.cam is None or self.IsGrabbing():
            return None

        res: Optional[pylon.GrabResult] = self.cam.GrabOne(1000)
        if res.GrabSucceeded():
            return res.GetArray()

    def RetrieveResult(self, timeout: int) -> Optional[pylon.GrabResult]:
        '''Retrieve the result of the grabbing

        Parameters
        ----------
        timeout : int
            The timeout for the grabbing in milliseconds

        Returns
        -------
        Optional[pylon.GrabResult]
            The result of the grabbing or None if failed
        '''
        try:
            return self.cam.RetrieveResult(timeout)
        except Exception as e:
            traceback.print_exc()
            return None
