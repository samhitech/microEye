from microEye.hardware.stages.kinesis.kdc101.enums import *


@dataclass
class KDC101Command:
    command_id: int
    data: bytes

    def to_bytes(self) -> bytes:
        return self.data


class KDC101_Factory:
    @staticmethod
    def create_command(
        cmd: KDC101_CMDS,
        channel: Union[int, Channels] = 0x00,
        param: Union[int, StopMode, JogDirection, ChannelEnableState] = 0x00,
        destination: int = 0x50,
        source: int = 0x01,
        **kwargs,
    ) -> KDC101Command:
        if not isinstance(cmd, KDC101_CMDS):
            raise ValueError('Invalid command')

        if isinstance(param, Enum) and isinstance(param.value, int):
            param = param.value
        elif not isinstance(param, int):
            param = 0x00

        if isinstance(channel, Enum) and isinstance(channel.value, int):
            channel = channel.value
        elif not isinstance(channel, int):
            channel = 0x00

        if cmd.cmd_type in [CommandType.CHANNEL, CommandType.COMPLEX]:
            KDC101_Factory._validate_channel(channel)

        if cmd.cmd_type == CommandType.GENERIC:
            channel = 0x00
            param = 0x00

        if cmd.cmd_type == CommandType.COMPLEX:
            data = KDC101_Factory._build_data(cmd.packet, kwargs)

            header = struct.pack(
                '<HHBBH', cmd.value, len(data) + 2, destination | 0x80, source, channel
            )

            header += data
        else:
            header = struct.pack(
                '<HBBBB', cmd.value, channel, param, destination, source
            )

        return KDC101Command(cmd.value, header)

    @staticmethod
    def _validate_channel(channel: int):
        try:
            return Channels(channel)
        except ValueError as e:
            raise ValueError('Invalid channel') from e

    @staticmethod
    def _build_data(structure: PacketStructure, kwargs: dict) -> bytes:
        format_string = '<'  # Little-endian
        values = []
        missing_fields = []

        for field in structure.data_fields:
            if field.name not in kwargs:
                missing_fields.append(field.name)

            value = kwargs.get(field.name, 0)

            if value is None:
                continue  # Skip optional fields that weren't provided

            if field.data_type == DataType.STRING:
                format_string += f'{field.length}s'
                values.append(value.encode('ascii'))
            else:
                format_string += field.data_type.value
                values.append(value)

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        return struct.pack(format_string, *values)

    @staticmethod
    def interpret_response(
        response_bytes: bytes,
    ) -> tuple[KDC101_RESPONSES, dict[str, Any]]:
        if len(response_bytes) < 6:
            raise ValueError('Response too short to be valid')

        # Extract the message ID from the first two bytes
        msg_id = struct.unpack('<H', response_bytes[:2])[0]

        # Find the corresponding response enum
        response_enum = KDC101_RESPONSES.get(msg_id)
        if response_enum is None:
            raise ValueError(f'Unknown response ID: 0x{msg_id:04X}')

        # Parse the response based on its structure
        parsed_data = {}
        if response_enum.packet:
            parsed_data = KDC101_Factory._parse_response_data(
                response_enum.packet, response_bytes
            )

        return response_enum, parsed_data

    @staticmethod
    def _parse_response_data(structure: PacketStructure, data: bytes) -> dict[str, Any]:
        parsed_data = {}
        offset = 0

        for field in structure.data_fields:
            if field.data_type == DataType.STRING:
                value = (
                    data[offset : offset + field.length].decode('ascii').rstrip('\x00')
                )
                offset += field.length
            else:
                format_char = field.data_type.value
                size = struct.calcsize(format_char)
                value = struct.unpack(f'<{format_char}', data[offset : offset + size])[
                    0
                ]
                offset += size

            parsed_data[field.name] = value

        return parsed_data

    @staticmethod
    def get_packet_size(packet: PacketStructure) -> int:
        size = 0
        for field in packet.data_fields:
            if field.data_type == DataType.STRING:
                size += field.length
            else:
                size += struct.calcsize(field.data_type.value)
        return size

    @staticmethod
    def get_response_size(msg_id: int) -> int:
        try:
            response_enum = KDC101_RESPONSES.get(msg_id)
            if response_enum.packet:
                return KDC101_Factory.get_packet_size(response_enum.packet)
            else:
                return 6  # Minimum message size
        except ValueError:
            return 6  # Unknown message, assume minimum size

    @staticmethod
    def interpret_status_bits(status_bits: int) -> dict[str, bool]:
        status = {}
        for bit in MotorStatusBits:
            status[bit.name] = bool(status_bits & bit.value)
        return status


# if __name__ == '__main__':
#     print(KDC101_CMDS.MGMSG_MOD_IDENTIFY)
#     KDC101_CMDS.get([1096 & 0xFF, 1096 >> 8])
#     print(KDC101_CMDS.MGMSG_MOD_IDENTIFY)
