from collections import deque
from dataclasses import dataclass, field

from microEye.hardware.stages.kinesis.kdc101.factory import *
from microEye.qt import QtCore, QtSerialPort, Signal
from microEye.utils.thread_worker import *


@dataclass
class KDC101State:
    position: float = 0.0
    velocity: float = 0.0
    backlash: float = 0.0
    current: float = 0.0
    encoder_count: int = 0
    info_serial_number: int = 0
    info_model: str = ''
    info_type: int = 0
    info_minor_revision_number: int = 0
    info_interim_revision_number: int = 0
    info_major_revision_number: int = 0
    info_hw_version: int = 0
    info_mod_state: int = 0
    info_nchs: int = 0
    status_code: int = 0
    velocity_min_vel: float = 0.0
    velocity_accn: float = 0.0
    velocity_max_vel: float = 0.0
    jog_mode: int = 0
    jog_step: float = 0.0
    jog_min_vel: float = 0.0
    jog_accn: float = 0.0
    jog_max_vel: float = 0.0
    jog_stop_mode: int = 0
    home_direction: int = 0
    home_limit_switch: bool = False
    home_velocity: float = 0.0
    home_zero_offset: float = 0.0
    limit_switch_cw_hard_limit: int = 0
    limit_switch_ccw_hard_limit: int = 0
    limit_switch_cw_soft_limit: float = 0.0
    limit_switch_ccw_soft_limit: float = 0.0
    limit_switch_soft_limit_mode: int = 0
    response_msg_id : int = 0
    response_code : int = 0
    response_notes : str = ''


class KDC101Controller(QtCore.QObject):
    '''Class for controlling Thorlab Z825B 25mm actuator by a KDC101'''

    SOURCE = 0x01
    DESTINATION = 0x50
    ENCODER_COUNT = 34555
    SOFTLIMIT_FACTOR = 134218

    dataReceived = Signal(tuple)
    '''Signal emitted when data is received from the device'''
    onWrite = Signal(bytearray)
    '''Signal emitted when data is written to the device'''
    stateChanged = Signal(str, object)
    '''Signal for state changes'''

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.__serial = QtSerialPort.QSerialPort(None, readyRead=self.rx_slot)
        self.__serial.setBaudRate(115200)
        self.__serial.setDataBits(QtSerialPort.QSerialPort.DataBits.Data8)
        self.__serial.setParity(QtSerialPort.QSerialPort.Parity.NoParity)
        self.__serial.setFlowControl(
            QtSerialPort.QSerialPort.FlowControl.HardwareControl
        )
        self.__serial.setPortName(kwargs.get('portName', 'COM1'))

        self.__buffer = []
        self.__responses = deque(maxlen=256)

        self.__channel = kwargs.get('channel', Channels.NA)

        self.onWrite.connect(lambda data: self.__serial.write(data))

        self.__state = KDC101State()

    @property
    def state(self) -> KDC101State:
        return self.__state

    @property
    def position(self) -> float:
        return self.__state.position

    def _update_state(self, attr: str, value: object):
        setattr(self.__state, attr, value)
        self.stateChanged.emit(attr, value)

    def _handle_state_update(self, response_enum: KDC101_RESPONSES, parsed_data: dict):
        if response_enum in [
            KDC101_RESPONSES.MGMSG_MOT_MOVE_COMPLETED,
            KDC101_RESPONSES.MGMSG_MOT_MOVE_STOPPED,
            KDC101_RESPONSES.MGMSG_MOT_GET_DCSTATUSUPDATE,
        ]:
            self._update_state('position', parsed_data['position'] / self.ENCODER_COUNT)
            self._update_state('velocity', parsed_data['velocity'] / self.ENCODER_COUNT)
            self._update_state(
                'current',
                parsed_data['motor current'] / 1000,
            )
            self._update_state(
                'status_code',
                parsed_data['status bits'],
            )
        elif response_enum == KDC101_RESPONSES.MGMSG_MOT_GET_VELPARAMS:
            self._update_state(
                'velocity_min_vel', parsed_data['min velocity'] / self.ENCODER_COUNT
            )
            self._update_state(
                'velocity_accn', parsed_data['acceleration'] / self.ENCODER_COUNT
            )
            self._update_state(
                'velocity_max_vel', parsed_data['max velocity'] / self.ENCODER_COUNT
            )
        elif response_enum == KDC101_RESPONSES.MGMSG_MOT_GET_JOGPARAMS:
            self._update_state('jog_mode', JogMode(parsed_data['jog mode']))
            self._update_state(
                'jog_step', parsed_data['jog step size'] / self.ENCODER_COUNT
            )
            self._update_state(
                'jog_min_vel', parsed_data['jog min velocity'] / self.ENCODER_COUNT
            )
            self._update_state(
                'jog_accn', parsed_data['jog acceleration'] / self.ENCODER_COUNT
            )
            self._update_state(
                'jog_max_vel', parsed_data['jog max velocity'] / self.ENCODER_COUNT
            )
            self._update_state('jog_stop_mode', StopMode(parsed_data['jog stop mode']))
        elif response_enum == KDC101_RESPONSES.MGMSG_MOT_GET_HOMEPARAMS:
            self._update_state('home_direction', parsed_data['direction'])
            self._update_state('home_limit_switch', parsed_data['limit switch'])
            self._update_state(
                'home_velocity', parsed_data['velocity'] / self.ENCODER_COUNT
            )
            self._update_state(
                'home_zero_offset', parsed_data['zero offset'] / self.ENCODER_COUNT
            )
        elif response_enum == KDC101_RESPONSES.MGMSG_MOT_GET_LIMSWITCHPARAMS:
            self._update_state(
                'limit_switch_cw_hard_limit',
                LimitSwitchOperation(parsed_data['cw hard limit']),
            )
            self._update_state(
                'limit_switch_ccw_hard_limit',
                LimitSwitchOperation(parsed_data['ccw hard limit']),
            )
            self._update_state(
                'limit_switch_cw_soft_limit',
                parsed_data['cw soft limit'] / self.SOFTLIMIT_FACTOR,
            )
            self._update_state(
                'limit_switch_ccw_soft_limit',
                parsed_data['ccw soft limit'] / self.SOFTLIMIT_FACTOR,
            )
            self._update_state(
                'limit_switch_soft_limit_mode',
                LimitSwitchMode(parsed_data['soft limit mode']),
            )
        elif response_enum == KDC101_RESPONSES.MGMSG_MOT_GET_ENCCOUNTER:
            self._update_state('encoder_count', parsed_data['encoder count'])
        elif response_enum == KDC101_RESPONSES.MGMSG_MOT_GET_STATUSBITS:
            self._update_state('status_code', parsed_data['status bits'])
        elif response_enum == KDC101_RESPONSES.MGMSG_HW_GET_INFO:
            self._update_state('info_serial_number', parsed_data['serial number'])
            self._update_state('info_model', parsed_data['model'])
            self._update_state('info_type', parsed_data['type'])
            self._update_state(
                'info_minor_revision_number', parsed_data['minor revision number']
            )
            self._update_state(
                'info_interim_revision_number', parsed_data['interim revision number']
            )
            self._update_state(
                'info_major_revision_number', parsed_data['major revision number']
            )
            self._update_state('info_hw_version', parsed_data['hw version'])
            self._update_state('info_mod_state', parsed_data['mod state'])
            self._update_state('info_nchs', parsed_data['nchs'])
        elif response_enum == KDC101_RESPONSES.MGMSG_MOT_GET_GENMOVEPARAMS:
            self._update_state('backlash', parsed_data['distance'] / self.ENCODER_COUNT)
        elif response_enum == KDC101_RESPONSES.MGMSG_MOT_GET_POSCOUNTER:
            self._update_state('position', parsed_data['position'] / self.ENCODER_COUNT)
        elif response_enum == KDC101_RESPONSES.MGMSG_HW_RESPONSE:
            self._update_state('response_code', parsed_data['code'])
        elif response_enum == KDC101_RESPONSES.MGMSG_HW_RICHRESPONSE:
            self._update_state('response_msg_id', parsed_data['msg_id'])
            self._update_state('response_code', parsed_data['code'])
            self._update_state('response_notes', parsed_data['notes'])
        elif response_enum == KDC101_RESPONSES.MGMSG_MOT_MOVE_HOMED:
            self._update_state('position', 0)

    @property
    def source(self):
        return self.SOURCE

    @property
    def destination(self):
        return self.DESTINATION

    @property
    def channel(self):
        return self.__channel

    def open(self, portName: str = None):
        '''
        Open the serial port.
        '''
        if portName:
            self.__serial.setPortName(portName)

        if not self.isOpen():
            self.__serial.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)

            self.resume_end_of_move_msgs()

            self.get_position()

        return self.isOpen()

    def close(self):
        '''
        Close the serial port.
        '''
        if self.isOpen():
            self.__serial.close()

    def isOpen(self):
        '''
        Check if the serial port is open.

        Returns
        -------
        bool
            True if the serial port is open, False otherwise.
        '''
        return self.__serial.isOpen()

    def portName(self):
        '''
        Get the name of the serial port.

        Returns
        -------
        str
            The name of the serial port.
        '''
        return self.__serial.portName()

    def setPortName(self, name: str):
        '''
        Set the name of the serial port.

        Parameters
        ----------
        name : str
            The new port name.
        '''
        if not self.isOpen():
            self.__serial.setPortName(name)

    def write(self, cmd: KDC101Command) -> Optional[tuple[KDC101_RESPONSES, dict]]:
        self.onWrite.emit(bytearray(cmd.data))

        response = KDC101_CMDS.get(cmd.command_id).response
        if response:
            return self.wait_for_response(response)

    def identify(self):
        if self.isOpen():
            _IDENTIFY = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOD_IDENTIFY,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_IDENTIFY)

    def home(self):
        if self.isOpen():
            _HOME = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_MOVE_HOME,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_HOME)

    def jog_fw(self):
        if self.isOpen():
            _JOG_FW = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_MOVE_JOG,
                channel=self.__channel,
                param=JogDirection.FORWARD,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_JOG_FW)

    def jog_bw(self):
        if self.isOpen():
            _JOG_BW = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_MOVE_JOG,
                channel=self.__channel,
                param=JogDirection.REVERSE,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_JOG_BW)

    def to_encoder(self, value: float):
        return int(self.ENCODER_COUNT * value)

    def move_absolute(self, position=0.1):
        if self.isOpen():
            _position = self.to_encoder(position)
            _ABSOLUTE = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_MOVE_ABSOLUTE,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
                position=_position,
            )
            return self.write(_ABSOLUTE)

    def move_relative(self, distance=0.1):
        if self.isOpen():
            _distance = self.to_encoder(distance)
            _RELATIVE = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_MOVE_RELATIVE,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
                distance=_distance,
            )
            return self.write(_RELATIVE)

    def move_stop(self, mode=StopMode.IMMEDIATE):
        if self.isOpen():
            _STOP = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_MOVE_STOP,
                channel=self.__channel,
                param=mode,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_STOP)

    def start_updates(self):
        if self.isOpen():
            _START_UPDATES = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_HW_START_UPDATEMSGS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_START_UPDATES)

    def stop_updates(self):
        if self.isOpen():
            _STOP_UPDATES = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_HW_STOP_UPDATEMSGS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_STOP_UPDATES)

    def resume_end_of_move_msgs(self):
        if self.isOpen():
            _RESUME_EOM = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_RESUME_ENDOFMOVEMSGS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_RESUME_EOM)

    def suspend_end_of_move_msgs(self):
        if self.isOpen():
            _STOP_EOM = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_SUSPEND_ENDOFMOVEMSGS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_STOP_EOM)

    def get_info(self):
        if self.isOpen():
            _REQ_INFO = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_HW_REQ_INFO,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            response = self.write(_REQ_INFO)
            if response:
                _, parsed_data = response
                return parsed_data

    def get_position(self):
        if self.isOpen():
            _REQ_POS = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_REQ_POSCOUNTER,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            response = self.write(_REQ_POS)
            if response:
                _, parsed_data = response
                return parsed_data['position'] / self.ENCODER_COUNT
        return None

    def get_encoder_count(self):
        if self.isOpen():
            _REQ_ENC = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_REQ_ENCCOUNTER,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            response = self.write(_REQ_ENC)
            if response:
                _, parsed_data = response
                return parsed_data['encoder count']
        return None

    def get_status(self):
        if self.isOpen():
            _REQ_STATUS = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_REQ_STATUSBITS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            response = self.write(_REQ_STATUS)
            if response:
                _, parsed_data = response
                return KDC101_Factory.interpret_status_bits(parsed_data['status_bits'])
        return None

    def get_status_update(self):
        if self.isOpen():
            _REQ_STATUS = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_REQ_DCSTATUSUPDATE,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_REQ_STATUS)

    def get_velocity_params(self):
        if self.isOpen():
            _REQ_VEL = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_REQ_VELPARAMS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_REQ_VEL)

    def get_jog_params(self):
        if self.isOpen():
            _REQ_JOG = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_REQ_JOGPARAMS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_REQ_JOG)

    def set_velocity_params(self, min_vel=0.0, accn=0.0, max_vel=0.0):
        if self.isOpen():
            return
            _min_vel = self.to_encoder(min_vel)
            _accn = self.to_encoder(accn)
            _max_vel = self.to_encoder(max_vel)
            _SET_VEL = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_SET_VELPARAMS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
                min_velocity=_min_vel,
                acceleration=_accn,
                max_velocity=_max_vel,
            )
            return self.write(_SET_VEL)

    def set_jog_params(
        self,
        mode=JogMode.SINGLE_STEP,
        step=0.1,
        min_vel=0.0,
        accn=0.0,
        max_vel=1.0,
        stop_mode=StopMode.PROFILED,
    ):
        if self.isOpen():
            return
            _mode = mode.value
            _step = self.to_encoder(step)
            _min_vel = self.to_encoder(min_vel)
            _accn = self.to_encoder(accn)
            _max_vel = self.to_encoder(max_vel)
            _stop_mode = stop_mode.value
            _SET_JOG = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_SET_JOGPARAMS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
                jog_mode=_mode,
                jog_step_size=_step,
                jog_min_velocity=_min_vel,
                jog_acceleration=_accn,
                jog_max_velocity=_max_vel,
                jog_stop_mode=_stop_mode,
            )
            return self.write(_SET_JOG)

    def get_backlash(self):
        if self.isOpen():
            _REQ_BACKLASH = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_REQ_GENMOVEPARAMS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_REQ_BACKLASH)

    def set_backlash(self, backlash=0.0):
        if self.isOpen():
            _backlash = self.to_encoder(backlash)
            _SET_BACKLASH = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_SET_GENMOVEPARAMS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
                distance=_backlash,
            )
            return self.write(_SET_BACKLASH)

    def get_limit_switch_params(self):
        if self.isOpen():
            _REQ_LIMIT = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_REQ_LIMSWITCHPARAMS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_REQ_LIMIT)

    def set_limit_switch_params(
        self,
        cw_hard_limit: LimitSwitchOperation,
        ccw_hard_limit: LimitSwitchOperation,
        cw_soft_limit: float,
        ccw_soft_limit: float,
        soft_limit_mode: LimitSwitchMode,
    ):
        if self.isOpen():
            return
            cw_hard_limit = cw_hard_limit.value
            ccw_hard_limit = ccw_hard_limit.value
            cw_soft_limit *= self.SOFTLIMIT_FACTOR
            ccw_soft_limit *= self.SOFTLIMIT_FACTOR
            soft_limit_mode = soft_limit_mode.value
            _SET_LIMIT = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_SET_LIMSWITCHPARAMS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
                cw_hard_limit=cw_hard_limit,
                ccw_hard_limit=ccw_hard_limit,
                cw_soft_limit=cw_soft_limit,
                ccw_soft_limit=ccw_soft_limit,
                soft_limit_mode=soft_limit_mode,
            )
            return self.write(_SET_LIMIT)

    def get_home_params(self):
        if self.isOpen():
            _REQ_HOME = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_REQ_HOMEPARAMS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
            )
            return self.write(_REQ_HOME)

    def set_home_params(
        self,
        direction: JogDirection,
        limit_switch: bool,
        velocity: float,
        zero_offset: float,
    ):
        if self.isOpen():
            return
            direction = direction.value
            limit_switch = 0x01 if limit_switch else 0x02
            velocity = self.to_encoder(velocity)
            zero_offset = self.to_encoder(zero_offset)
            _SET_HOME = KDC101_Factory.create_command(
                KDC101_CMDS.MGMSG_MOT_SET_HOMEPARAMS,
                channel=self.__channel,
                destination=self.destination,
                source=self.source,
                direction=direction,
                lim_switch=limit_switch,
                velocity=velocity,
                zero_offset=zero_offset,
            )
            return self.write(_SET_HOME)

    def rx_slot(self):
        '''
        Handle incoming data from the serial port.
        '''
        try:
            data = self.__serial.readAll().data()
            self.__buffer.extend(data)

            while len(self.__buffer) >= 6:
                msg_id = struct.unpack('<H', bytes(self.__buffer[:2]))[0]

                msg_size = KDC101_Factory.get_response_size(msg_id)

                if len(self.__buffer) < msg_size:
                    break  # Wait for more data

                message = bytes(self.__buffer[:msg_size])
                self.__buffer = self.__buffer[msg_size:]

                try:
                    response_enum, parsed_data = KDC101_Factory.interpret_response(
                        message
                    )
                    print(response_enum, parsed_data)
                    self.__responses.append((response_enum, parsed_data))
                    self.dataReceived.emit((response_enum, parsed_data))
                    self._handle_state_update(response_enum, parsed_data)
                except ValueError as e:
                    print(f'Error interpreting response: {e}')

        except Exception as e:
            self.__serial.errorOccurred.emit(str(e))

    def clearResponses(self):
        '''
        Clear the response buffer.
        '''
        self.__responses.clear()
        self.__buffer = []

    def wait_for_response(
        self, expected_response: KDC101_RESPONSES, timeout: int = 60000
    ) -> Optional[tuple[KDC101_RESPONSES, dict]]:
        start_time = QtCore.QDateTime.currentMSecsSinceEpoch()

        while QtCore.QDateTime.currentMSecsSinceEpoch() - start_time < timeout:
            if self.__responses:
                response_enum, parsed_data = self.__responses.popleft()
                if response_enum == expected_response:
                    return response_enum, parsed_data

            QtCore.QCoreApplication.processEvents()
            QtCore.QThread.msleep(10)

        print(f'Timeout waiting for response: {expected_response}')
        return None
