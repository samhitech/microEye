import struct
from dataclasses import dataclass
from enum import Enum, IntFlag, auto
from typing import Any, Optional, Union


class CommandType(Enum):
    GENERIC = auto()
    CHANNEL = auto()
    COMPLEX = auto()


class DataType(Enum):
    WORD = 'H'  # Unsigned 16-bit integer
    SHORT = 'h'  # Signed 16-bit integer
    DWORD = 'I'  # Unsigned 32-bit integer
    LONG = 'i'  # Signed 32-bit integer
    CHAR = 'B'  # 1 byte
    STRING = 's'  # String of N characters


@dataclass
class DataField:
    name: str
    data_type: DataType
    length: int = 1  # For strings, specify the number of characters


@dataclass
class PacketStructure:
    data_fields: list[DataField]

    @property
    def has_data(self) -> bool:
        return len(self.data_fields) > 0

    @property
    def has_length(self) -> bool:
        return any(field.name == 'length' for field in self.data_fields)


class KDC101_RESPONSES(Enum):
    def __init__(self, value, packet: Optional[PacketStructure] = None):
        self._value_ = value
        self.packet = packet


    MGMSG_MOD_GET_CHANENABLESTATE = (
        0x0212,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('channel', DataType.CHAR),
                DataField('enable_state', DataType.CHAR),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
            ]
        ),
    )
    '''Response of MGMSG_MOD_REQ_CHANENABLESTATE'''

    MGMSG_HW_RESPONSE = (
        0x0080,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('code', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
            ]
        ),
    )
    '''Sent by the controllers to notify Thorlabs Server of some event that
    requires user intervention, usually some fault or error condition that
    needs to be handled before normal operation can resume.

    The message transmits the fault code as a numerical value – see the
    Return Codes listed in the Thorlabs Server helpfile for details on the
    specific return codes.
    '''
    MGMSG_HW_RICHRESPONSE = (
        0x0081,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('msg_id', DataType.WORD),
                DataField('code', DataType.WORD),
                DataField('notes', DataType.STRING, length=64),
            ]
        ),
    )
    '''
    Similarly, to HW_RESPONSE, this message is sent by the controllers
    to notify Thorlabs Server of some event that requires user
    intervention,

    usually some fault or error condition that needs to be
    handled before normal operation can resume.

    However, unlike HW_RESPONSE, this message also transmits a printable text string.

    Upon receiving the message, Thorlabs Server displays both the
    numerical value and the text information, which is useful in finding
    the cause of the problem.
    '''

    MGMSG_HW_GET_INFO = (
        0x0006,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('serial number', DataType.LONG),
                DataField('model', DataType.STRING, length=8),
                DataField('type', DataType.WORD),
                DataField('minor revision number', DataType.CHAR),
                DataField('interim revision number', DataType.CHAR),
                DataField('major revision number', DataType.CHAR),
                DataField('unused', DataType.CHAR),
                DataField('hw version', DataType.WORD),
                DataField('mod state', DataType.WORD),
                DataField('nchs', DataType.WORD),
            ]
        ),
    )
    '''Response of MGMSG_HW_REQ_INFO'''

    MGMSG_HUB_GET_BAYUSED = (
        0x0066,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('bay id', DataType.CHAR),
                DataField('unused', DataType.CHAR),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
            ]
        ),
    )
    '''Response of MGMSG_HUB_REQ_BAYUSED'''

    MGMSG_MOT_GET_POSCOUNTER = (
        0x0412,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('channel', DataType.WORD),
                DataField('position', DataType.LONG),
            ]
        ),
    )
    '''Response of MGMSG_MOT_REQ_POSCOUNTER'''

    MGMSG_MOT_GET_ENCCOUNTER = (
        0x040B,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('channel', DataType.WORD),
                DataField('encoder count', DataType.LONG),
            ]
        ),
    )
    '''Response of MGMSG_MOT_REQ_ENCCOUNTER'''

    MGMSG_MOT_GET_VELPARAMS = (
        0x0415,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('channel', DataType.WORD),
                DataField('min velocity', DataType.LONG),
                DataField('acceleration', DataType.LONG),
                DataField('max velocity', DataType.LONG),
            ]
        ),
    )
    '''Response of MGMSG_MOT_REQ_VELPARAMS'''

    MGMSG_MOT_GET_JOGPARAMS = (
        0x0418,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('channel', DataType.WORD),
                DataField('jog mode', DataType.WORD),
                DataField('jog step size', DataType.LONG),
                DataField('jog min velocity', DataType.LONG),
                DataField('jog acceleration', DataType.LONG),
                DataField('jog max velocity', DataType.LONG),
                DataField('jog stop mode', DataType.WORD),
            ]
        ),
    )
    '''Response of MGMSG_MOT_REQ_JOGPARAMS'''

    MGMSG_MOT_GET_LIMSWITCHPARAMS = (
        0x0425,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('channel', DataType.WORD),
                DataField('cw hard limit', DataType.WORD),
                DataField('ccw hard limit', DataType.WORD),
                DataField('cw soft limit', DataType.LONG),
                DataField('ccw soft limit', DataType.LONG),
                DataField('soft limit mode', DataType.WORD),
            ]
        ),
    )
    '''Response of MGMSG_MOT_REQ_LIMSWITCHPARAMS'''

    MGMSG_MOT_GET_GENMOVEPARAMS = (
        0x043C,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('channel', DataType.WORD),
                DataField('distance', DataType.LONG),
            ]
        ),
    )
    '''Response of MGMSG_MOT_REQ_GENMOVEPARAMS'''

    MGMSG_MOT_GET_HOMEPARAMS = (
        0x0442,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('channel', DataType.WORD),
                DataField('direction', DataType.WORD),
                DataField('limit switch', DataType.WORD),
                DataField('velocity', DataType.LONG),
                DataField('zero offset', DataType.LONG),
            ]
        ),
    )
    '''Response of MGMSG_MOT_REQ_HOMEPARAMS'''

    MGMSG_MOT_MOVE_HOMED = (
        0x0444,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('channel', DataType.CHAR),
                DataField('unused', DataType.CHAR),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
            ]
        ),
    )
    '''
    No response on initial message, but upon completion of home
    sequence controller sends a “homing completed” message'''

    MGMSG_MOT_GET_MOVERELPARAMS = (
        0x0447,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('channel', DataType.WORD),
                DataField('distance', DataType.LONG),
            ]
        ),
    )
    '''Response of MGMSG_MOT_REQ_MOVERELPARAMS'''

    MGMSG_MOT_GET_MOVEABSPARAMS = (
        0x0452,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('channel', DataType.WORD),
                DataField('position', DataType.LONG),
            ]
        ),
    )
    '''Response of MGMSG_MOT_REQ_MOVEABSPARAMS'''

    MGMSG_MOT_MOVE_COMPLETED = (
        0x0464,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('channel', DataType.CHAR),
                DataField('unused', DataType.CHAR),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                # MOT DC STATUS
                DataField('channel', DataType.WORD),
                DataField('position', DataType.LONG),
                DataField('velocity', DataType.WORD),
                DataField('motor current', DataType.WORD),
                DataField('status bits', DataType.DWORD),
            ]
        ),
    )
    '''
    No response on initial message, but upon completion of the relative
    or absolute move sequence, the controller sends a “move
    completed” message.
    '''

    MGMSG_MOT_MOVE_STOPPED = (
        0x0466,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                # MOT DC STATUS
                DataField('channel', DataType.WORD),
                DataField('position', DataType.LONG),
                DataField('velocity', DataType.WORD),
                DataField('motor current', DataType.WORD),
                DataField('status bits', DataType.DWORD),
            ]
        ),
    )
    '''
    No response on initial message, but upon completion of the stop
    move, the controller sends a “move stopped” message.
    '''

    MGMSG_MOT_GET_DCSTATUSUPDATE = (
        0x0491,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                # MOT DC STATUS
                DataField('channel', DataType.WORD),
                DataField('position', DataType.LONG),
                DataField('velocity', DataType.WORD),
                DataField('motor current', DataType.WORD),
                DataField('status bits', DataType.DWORD),
            ]
        ),
    )
    '''Response of MGMSG_MOT_REQ_DCSTATUSUPDATE'''

    # MGMSG_MOT_ACK_DCSTATUSUPDATE = 0x0492
    '''Acknowledgement of MGMSG_MOT_REQ_DCSTATUSUPDATE

    Only Applicable If Using USB COMMS. Does not apply to RS-232 COMMS'''

    MGMSG_MOT_GET_STATUSBITS = (
        0x042A,
        PacketStructure(
            [
                DataField('id', DataType.WORD),
                DataField('length', DataType.WORD),
                DataField('destination', DataType.CHAR),
                DataField('source', DataType.CHAR),
                DataField('channel', DataType.WORD),
                DataField('status bits', DataType.DWORD),
            ]
        ),
    )
    '''Response of MGMSG_MOT_REQ_STATUSBITS'''

    @classmethod
    def get(cls, response: Union[int, list]):
        '''Get the response enum for a given response code.'''
        try:
            if isinstance(response, list):
                if len(response) < 2:
                    raise ValueError('Not enough bytes to get command.')

                response = (response[1] << 8) | response[0]

            if not isinstance(response, int):
                raise ValueError(
                    'Command must be an integer or list of integers/bytes.'
                )

            # Iterate through enum members to find the one with the matching value
            for member in cls:
                if member.value == response:
                    return member
        except ValueError:
            return None


class KDC101_CMDS(Enum):
    def __init__(
        self,
        value,
        cmd_type: CommandType,
        packet: Optional[PacketStructure] = None,
        response: Optional[KDC101_RESPONSES] = None,
    ):
        self._value_ = value
        self.cmd_type = cmd_type
        self.packet = packet
        self.response = response

    MGMSG_MOD_IDENTIFY = (0x0223, CommandType.CHANNEL)
    '''Instruct hardware unit to identify itself (by flashing its front panel
    LEDs).'''

    MGMSG_MOD_SET_CHANENABLESTATE = (0x0210, CommandType.CHANNEL)
    '''Sent to enable or disable the specified drive channel.'''
    MGMSG_MOD_REQ_CHANENABLESTATE = (
        0x0211,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOD_GET_CHANENABLESTATE,
    )
    '''Sent to request the enable state of the specified drive channel.'''

    MGMSG_HW_DISCONNECT = (0x0002, CommandType.GENERIC)
    '''Sent by the hardware unit or host to disconnect from the
    Ethernet/USB bus.'''

    MGMSG_HW_START_UPDATEMSGS = (0x0011, CommandType.GENERIC)
    '''Sent to start automatic status updates from the embedded controller.'''
    MGMSG_HW_STOP_UPDATEMSGS = (0x0012, CommandType.GENERIC)
    '''
    Sent to stop automatic status updates from the controller – usually
    called by a client application when it is shutting down, to instruct
    the controller to turn off status updates to prevent USB buffer
    overflows on the PC.
    '''

    MGMSG_HW_REQ_INFO = (
        0x0005,
        CommandType.GENERIC,
        None,
        KDC101_RESPONSES.MGMSG_HW_GET_INFO,
    )
    '''Sent to request hardware information from the controller.'''

    MGMSG_HUB_REQ_BAYUSED = (
        0x0065,
        CommandType.GENERIC,
        None,
        KDC101_RESPONSES.MGMSG_HUB_GET_BAYUSED,
    )
    '''Sent to determine to which bay a specific unit is fitted.'''

    MGMSG_MOT_REQ_POSCOUNTER = (
        0x0411,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_GET_POSCOUNTER,
    )
    '''Used to request the ‘live’ position count in the controller.'''

    MGMSG_MOT_REQ_ENCCOUNTER = (
        0x040A,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_GET_ENCCOUNTER,
    )
    '''
    Used to set the encoder count in the controller and is
    only applicable to stages and actuators fitted with an encoder.
    '''

    MGMSG_MOT_SET_VELPARAMS = (
        0x0413,
        CommandType.COMPLEX,
        PacketStructure(
            [
                DataField('min_velocity', DataType.LONG),
                DataField('acceleration', DataType.LONG),
                DataField('max_velocity', DataType.LONG),
            ]
        ),
    )
    '''
    Used to set the trapezoidal velocity parameters for the specified
    motor channel.

    For DC servo controllers, the velocity is set in
    encoder counts/sec and acceleration is set in encoder
    counts/sec/sec.

    For stepper motor controllers the velocity is set in microsteps/sec
    and acceleration is set in microsteps/sec/sec.'''
    MGMSG_MOT_REQ_VELPARAMS = (
        0x0414,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_GET_VELPARAMS,
    )
    '''
    Used to request the trapezoidal velocity parameters for the specified motor channel.
    '''

    MGMSG_MOT_SET_JOGPARAMS = (
        0x0416,
        CommandType.COMPLEX,
        PacketStructure(
            [
                DataField('jog_mode', DataType.WORD),
                DataField('jog_step_size', DataType.LONG),
                DataField('jog_min_velocity', DataType.LONG),
                DataField('jog_acceleration', DataType.LONG),
                DataField('jog_max_velocity', DataType.LONG),
                DataField('jog_stop_mode', DataType.WORD),
            ]
        ),
    )
    '''
    Used to set the velocity jog parameters for the specified motor
    channel.

    For DC servo controllers, values set in encoder counts.

    For stepper motor controllers the values is set in microsteps.'''
    MGMSG_MOT_REQ_JOGPARAMS = (
        0x0417,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_GET_JOGPARAMS,
    )
    '''Used to request the velocity jog parameters for the specified motor channel.'''

    MGMSG_MOT_SET_LIMSWITCHPARAMS = (
        0x0423,
        CommandType.COMPLEX,
        PacketStructure(
            [
                DataField('cw_hard_limit', DataType.WORD),
                DataField('ccw_hard_limit', DataType.WORD),
                DataField('cw_soft_limit', DataType.LONG),
                DataField('ccw_soft_limit', DataType.LONG),
                DataField('soft_limit_mode', DataType.WORD),
            ]
        ),
    )
    '''
    Used to set the limit switch parameters for the specified motor channel.
    '''
    MGMSG_MOT_REQ_LIMSWITCHPARAMS = (
        0x0424,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_GET_LIMSWITCHPARAMS,
    )
    '''Used to request the limit switch parameters for the specified motor channel.'''

    MGMSG_MOT_SET_GENMOVEPARAMS = (
        0x043A,
        CommandType.COMPLEX,
        PacketStructure([DataField('distance', DataType.LONG)]),
    )
    '''
    Used to set the general move parameters for the specified motor channel.

    Currently this refers specifically to the backlash settings. '''
    MGMSG_MOT_REQ_GENMOVEPARAMS = (
        0x043B,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_GET_GENMOVEPARAMS,
    )
    '''Used to request the general move parameters for the specified motor channel.'''

    MGMSG_MOT_SET_HOMEPARAMS = (
        0x0440,
        CommandType.COMPLEX,
        PacketStructure(
            [
                DataField('direction', DataType.WORD),
                DataField('lim_switch', DataType.WORD),
                DataField('velocity', DataType.LONG),
                DataField('zero_offset', DataType.LONG),
            ]
        ),
    )
    '''
    Used to set the home parameters for the specified motor channel.

    These parameters are stage specific and for the MLS203 stage
    implementation the only parameter that can be changed is the
    homing velocity.'''

    MGMSG_MOT_REQ_HOMEPARAMS = (
        0x0441,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_GET_HOMEPARAMS,
    )
    '''
    Used to request the home parameters for the specified motor channel.'''

    MGMSG_MOT_SET_MOVERELPARAMS = (
        0x0445,
        CommandType.COMPLEX,
        PacketStructure([DataField('distance', DataType.LONG)]),
    )
    '''
    Used to set the relative move parameters for the specified motor channel.

    The only significant parameter currently is the relative move distance itself.

    This gets stored by the controller and is used the next time a relative move
    is initiated.'''
    MGMSG_MOT_REQ_MOVERELPARAMS = (
        0x0446,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_GET_MOVERELPARAMS,
    )
    '''Used to request the relative move parameters for the specified motor channel.'''

    MGMSG_MOT_SET_MOVEABSPARAMS = (
        0x0450,
        CommandType.COMPLEX,
        PacketStructure([DataField('position', DataType.LONG)]),
    )
    '''
    Used to set the absolute move parameters for the specified motor channel.

    The only significant parameter currently is the absolute
    move position itself.

    This gets stored by the controller and is used
    the next time an absolute move is initiated.
    '''
    MGMSG_MOT_REQ_MOVEABSPARAMS = (
        0x0451,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_GET_MOVEABSPARAMS,
    )
    '''Used to request the absolute move parameters for the specified motor channel.'''

    MGMSG_MOT_MOVE_HOME = (
        0x0443,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_MOVE_HOMED,
    )
    '''Sent to start a home move sequence on the specified motor channel.'''

    MGMSG_MOT_MOVE_RELATIVE = (
        0x0448,
        CommandType.COMPLEX,
        PacketStructure([DataField('distance', DataType.LONG)]),
        KDC101_RESPONSES.MGMSG_MOT_MOVE_COMPLETED,
    )
    '''
    This command can be used to start a relative move on the specified
    motor channel (using the relative move distance parameter above).

    There are two versions of this command: a shorter (6-byte header
    only) version and a longer (6 byte header plus 6 data bytes) version.

    When the first one is used, the relative distance parameter used for
    the move will be the parameter sent previously by a
    MGMSG_MOT_SET_MOVERELPARAMS command.

    If the longer version of the command is used, the relative distance
    is encoded in the data packet that follows the header.
    '''

    MGMSG_MOT_MOVE_ABSOLUTE = (
        0x0453,
        CommandType.COMPLEX,
        PacketStructure([DataField('position', DataType.LONG)]),
        KDC101_RESPONSES.MGMSG_MOT_MOVE_COMPLETED,
    )
    '''
    Used to start an absolute move on the specified motor channel
    (using the absolute move position parameter above).

    As previously described in the “MOVE RELATIVE” command, there are two
    versions of this command: a shorter (6-byte header only) version
    and a longer (6 byte header plus 6 data bytes) version.

    When the first one is used, the absolute move position parameter used for the
    move will be the parameter sent previously by a MGMSG_MOT_SET_MOVEABSPARAMS command.

    If the longer version of the command is used, the absolute position is encoded in
    the data packet that follows the header.

    Upon completion of the absolute move the controller sends a Move Completed message
    as previously described.
    '''

    MGMSG_MOT_MOVE_JOG = (
        0x046A,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_MOVE_COMPLETED,
    )
    '''Sent to start a jog move on the specified motor channel.

    Upon completion of the jog move the controller sends a Move Completed message as
    previously described.

    **Note:**
    The direction of the jog move is device dependent, i.e., on some devices jog forward
    may be towards the home position while on other devices it could be the opposite.'''

    MGMSG_MOT_MOVE_VELOCITY = (
        0x0457,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_MOVE_COMPLETED,
    )
    '''
    This command can be used to start a move on the specified motor
    channel.

    When this method is called, the motor will move continuously in the
    specified direction, using the velocity parameters set in the
    MGMSG_MOT_SET_VELPARAMS command until either a stop
    command (either StopImmediate or StopProfiled) is called, or a limit
    switch is reached.
    '''

    MGMSG_MOT_MOVE_STOP = (
        0x0465,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_MOVE_STOPPED,
    )
    '''
    Sent to stop any type of motor move (relative, absolute, homing or move at velocity)
    on the specified motor channel.
    '''

    MGMSG_MOT_REQ_DCSTATUSUPDATE = (
        0x0490,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_GET_DCSTATUSUPDATE,
    )
    '''
    Used to request a status update for the specified motor channel.

    This request can be used instead of enabling regular updates as
    described above.'''

    MGMSG_MOT_REQ_STATUSBITS = (
        0x0429,
        CommandType.CHANNEL,
        None,
        KDC101_RESPONSES.MGMSG_MOT_GET_STATUSBITS,
    )
    '''
    Used to request a “cut down” version of the status update message,
    only containing the status bits, without data about position and
    velocity.
    '''

    MGMSG_MOT_SUSPEND_ENDOFMOVEMSGS = (0x046B, CommandType.GENERIC)
    '''
    Sent to disable all unsolicited end of move messages and error
    messages returned by the controller, i.e.

    - MGMSG_MOT_MOVE_STOPPED
    - MGMSG_MOT_MOVE_COMPLETED
    - MGMSG_MOT_MOVE_HOMED
    '''

    MGMSG_MOT_RESUME_ENDOFMOVEMSGS = (0x046C, CommandType.GENERIC)
    '''
    Sent to resume all unsolicited end of move messages and error
    messages returned by the controller, i.e.

    - MGMSG_MOT_MOVE_STOPPED
    - MGMSG_MOT_MOVE_COMPLETED
    - MGMSG_MOT_MOVE_HOMED

    The command also disables the error messages that the controller
    sends when an error conditions is detected:

    - MGMSG_HW_RESPONSE
    - MGMSG_HW_RICHRESPONSE

    This is the default state when the controller is powered up.'''

    @classmethod
    def get(cls, cmd: Union[int, list]):
        '''Get the command enum for a given command code.'''
        try:
            if isinstance(cmd, list):
                if len(cmd) < 2:
                    raise ValueError('Not enough bytes to get command.')

                cmd = (cmd[1] << 8) | cmd[0]

            if not isinstance(cmd, int):
                raise ValueError(
                    'Command must be an integer or list of integers/bytes.'
                )

            # Iterate through enum members to find the one with the matching value
            for member in cls:
                if member.value == cmd:
                    return member
        except ValueError:
            return None


class Channels(Enum):
    NA = 0x00
    CHAN1 = 0x01
    CHAN2 = 0x02
    CHAN3 = 0x04
    CHAN4 = 0x08


class ChannelEnableState(Enum):
    ENABLE = 0x01
    DISABLE = 0x02


class BayStates(Enum):
    BAY_OCCUPIED = 0x01
    BAY_EMPTY = 0x02


class HubBayIdents(Enum):
    STANDALONE = 0x01  # T-Cube being standalone, i.e., off the hub
    UNKNOWN = 0x00  # T-Cube on hub, but bay unknown
    BAY_1 = 0x01  # Bay 1
    BAY_2 = 0x02  # Bay 2
    BAY_3 = 0x03  # Bay 3
    BAY_4 = 0x04  # Bay 4
    BAY_5 = 0x05  # Bay 5
    BAY_6 = 0x06  # Bay 6


class JogDirection(Enum):
    FORWARD = 0x01  # Jog forward
    REVERSE = 0x02  # Jog in the reverse direction


class JogMode(Enum):
    CONTINUOUS = 0x01  # Continuous jogging
    SINGLE_STEP = 0x02  # Single step jogging


class StopMode(Enum):
    IMMEDIATE = 0x01  # Stop immediately (abrupt)
    PROFILED = 0x02  # Stop in a controller (profiled) manner


class MotorStatusBits(Enum):
    P_MOT_SB_CWHARDLIMIT = 0x00000001
    '''
    Clockwise hardware limit switch.

    On linear stages, this corresponds to the forward limit switch.'''

    P_MOT_SB_CCWHARDLIMIT = 0x00000002
    '''
    Counter-clockwise hardware limit switch.

    On linear stages, this corresponds to the reverse limit switch.'''

    P_MOT_SB_CWSOFTLIMIT = 0x00000004
    '''
    Clockwise software limit switch.

    Restricts motion to a narrower range than the hardware limit switch.'''

    P_MOT_SB_CCWSOFTLIMIT = 0x00000008
    '''
    Counter-clockwise software limit switch.

    Restricts motion to a narrower range than the hardware limit switch.'''

    P_MOT_SB_INMOTIONCW = 0x00000010
    '''Indicates that the motor is in motion, moving clockwise.'''

    P_MOT_SB_INMOTIONCCW = 0x00000020
    '''Indicates that the motor is in motion, moving counter-clockwise.'''

    P_MOT_SB_JOGGINGCW = 0x00000040
    '''Indicates that the motor is jogging, moving clockwise.'''

    P_MOT_SB_JOGGINGCCW = 0x00000080
    '''Indicates that the motor is jogging, moving counter-clockwise.'''

    P_MOT_SB_CONNECTED = 0x00000100
    '''Indicates that the motor has been recognized by the controller.'''

    P_MOT_SB_HOMING = 0x00000200
    '''Indicates that the motor is performing a homing move.'''

    P_MOT_SB_HOMED = 0x00000400
    '''Indicates that the motor has completed the homing move,
    and the absolute position is known.'''

    P_MOT_SB_POSITIONERROR = 0x00004000
    '''
    Indicates that the actual position is outside the margin specified around the
    trajectory position.

    (In simple terms the motor is not where it should be.)

    This can occur momentarily during fast acceleration
    (the motor lags behind the trajectory) or when the motor
    is jammed, or the move is obstructed.

    Typically the condition can trigger the controller to disable
    the motor in order to prevent damage, which in turn will clear the error.
    '''
    P_MOT_SB_DIGIP1 = 0x00100000
    P_MOT_SB_DIGIP2 = 0x00200000

    P_MOT_SB_ERROR = 0x40000000
    '''
    Indicates an error condition, either listed above or arising as a result of another
    abnormal condition.'''
    P_MOT_SB_ENABLED = 0x80000000
    '''
    Indicates that the motor output is enabled and the controller is in charge of
    maintaining the required position.

    When the output is disabled, the motor is not controlled by the electronics
    and can be moved manually, as much as the mechanical construction (such as
    any leadscrew and gearbox fitted) allows.

    This is not full list of all the bits but the remaining bits reflect information
    about the state of the hardware that in most cases does not affect motion.
    '''


class LimitSwitchOperation(IntFlag):
    '''Clockwise Limit switch operation flags.

    Flag SWAP_CW_CCW is required for CCW.'''

    IGNORE_SWITCH = 0x01
    '''Ignore switch or switch not present'''
    MAKES_ON_CONTACT = 0x02
    '''Switch makes on contact'''
    BREAKS_ON_CONTACT = 0x03
    '''Switch breaks on contact'''
    MAKES_ON_CONTACT_HOMING = 0x04
    '''Switch makes on contact (for homing)'''
    BREAKS_ON_CONTACT_HOMING = 0x05
    '''Switch breaks on contact (for homing)'''
    INDEX_MARK_HOMING = 0x06
    '''PMD based brushless servo controllers only, uses index mark for homing'''
    SWAP_CW_CCW = 0x80
    '''
    Bitwise OR this with one of the settings above to swap CW and CCW limit switches
    '''


class LimitSwitchMode(IntFlag):
    '''Software limit switch mode.

    Flag ROTATION_STAGE_LIMIT is required for rotation stage.'''

    IGNORE_LIMIT = 0x01
    '''Ignore limit'''
    STOP_IMMEDIATE = 0x02
    '''Stop immediately at limit'''
    PROFILED_STOP = 0x03
    '''Profiled stop at limit'''
    ROTATION_STAGE_LIMIT = 0x80
    '''
    Bitwise OR this with one of the settings
    above to set rotation stage limit.

    (Not applicable to TDC001 units)'''
