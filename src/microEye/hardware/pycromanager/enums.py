from enum import Enum


# ActionType
class ActionType(Enum):
    NoAction = 0
    BeforeGet = 1
    AfterSet = 2
    IsSequenceable = 3
    AfterLoadSequence = 4
    StartSequence = 5
    StopSequence = 6

# DeviceType
class DeviceType(Enum):
    UnknownType = 0
    AnyType = 1
    CameraDevice = 2
    ShutterDevice = 3
    StateDevice = 4
    StageDevice = 5
    XYStageDevice = 6
    SerialDevice = 7
    GenericDevice = 8
    AutoFocusDevice = 9
    CoreDevice = 10
    ImageProcessorDevice = 11
    SignalIODevice = 12
    MagnifierDevice = 13
    SLMDevice = 14
    HubDevice = 15
    GalvoDevice = 16

# PropertyType
class PropertyType(Enum):
    Undef = 0
    String = 1
    Float = 2
    Integer = 3

    def to_python(self):
        if self == PropertyType.String:
            return str
        elif self == PropertyType.Float:
            return float
        elif self == PropertyType.Integer:
            return int
        else:
            return None

    @classmethod
    def to_dict(cls):
        return {member.value: member for member in cls}

    @classmethod
    def from_java(cls, prop_jv):
        value = prop_jv.swig_value()
        members = cls.to_dict()
        if value in members:
            return members[value]
        else:
            raise ValueError('Invalid PropertyType value')

# PortType
class PortType(Enum):
    InvalidPort = 0
    SerialPort = 1
    USBPort = 2
    HIDPort = 3

# FocusDirection
class FocusDirection(Enum):
    Unknown = 0
    TowardSample = 1
    AwayFromSample = 2

# DeviceNotification
class DeviceNotification(Enum):
    Attention = 0
    Done = 1
    StatusChanged = 2

# DeviceDetectionStatus
class DeviceDetectionStatus(Enum):
    Unimplemented = -2
    Misconfigured = -1
    CanNotCommunicate = 0
    CanCommunicate = 1

# DeviceInitializationState
class DeviceInitializationState(Enum):
    Uninitialized = 0
    InitializedSuccessfully = 1
    InitializationFailed = 2
