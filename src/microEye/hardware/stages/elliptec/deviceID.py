import struct
from enum import Enum


class DeviceID:
    class DeviceTypes(Enum):
        '''Values that represent Elliptec Device Types.'''

        Paddle = 3
        '''OBSOLETE (ELL3)'''
        Rotator = 4
        '''OBSOLETE (ELL4)'''
        Actuator = 5
        '''OBSOLETE (ELL5)'''
        Shutter2 = 6
        '''An Elliptec Shutter/Slider SM1 x 2 (ELL6).'''
        LinearStage25mm = 7
        '''OBSOLETE: An Elliptec Linear Stage (ELL7).'''
        RotaryStage8 = 8
        '''OBSOLETE: An Elliptec Rotary Stage (ELL8).'''
        Shutter4 = 9
        '''An Elliptec Shutter/Slider SM1 x 4 (ELL9).'''
        LinearStage60mm_10 = 10
        '''OBSOLETE: An Elliptec Linear Stage (ELL10).'''
        Shutter6 = 12
        '''An Elliptec Shutter/Slider SM05 x 6 (ELL12).'''
        OpticsRotator = 14
        '''An Elliptec Optics Rotator (ELL14).'''
        MotorizedIris = 15
        '''An Elliptec Motorized Iris (ELL15).'''
        LinearStage28mm = 17
        '''An Elliptec 28mm Linear Stage (ELL17).'''
        RotaryStage18 = 18
        '''An Elliptec Rotary Stage (ELL18).'''
        LinearStage60mm = 20
        '''An Elliptec 60mm Linear Stage (ELL20).'''

        @property
        def string_value(self):
            values = {
                3: 'Paddle (OBSOLETE)',
                4: 'Rotator (OBSOLETE)',
                5: 'Actuator (OBSOLETE)',
                6: 'Slider SM1 x 2',
                7: 'Linear Stage 25mm (OBSOLETE)',
                8: 'Rotary Stage (OBSOLETE)',
                9: 'Slider SM1 x 4',
                10: 'Linear Stage 60mm (OBSOLETE)',
                12: 'Slider SM05 x 6',
                14: 'Optics Rotator',
                15: 'Motorized Iris',
                17: 'Linear Stage 28mm',
                18: 'Rotary Stage',
                20: 'Linear Stage 60mm',
            }
            return values[self.value]

    class UnitTypes(Enum):
        '''Values that represent units in the Elliptec devices.'''

        MM = 'mm'
        '''An enum constant representing the millimetres option.'''
        Inches = 'in'
        '''An enum constant representing the inches option.'''
        Degrees = 'deg'
        '''An enum constant representing the degrees option.'''

    def __init__(self, id_string=None):
        self._unit_factor = 0
        self._unit_type = None
        self.FormatStr = ''
        self.Units = ''

        if id_string:
            self._deserialize(id_string)

    def _deserialize(self, id_string):
        device_info_format = 'c2s2s8s4s2s2s4s8s'
        device_info = struct.unpack(device_info_format, id_string.encode())

        self.Address = device_info[0].decode()
        self.SerialNo = device_info[3].decode()
        self.Year = int(device_info[4])
        self.Firmware = float('.'.join(device_info[5].decode()))
        hw = int(device_info[6], 16)
        self.Hardware = hw & 0x7F
        self.Imperial = bool((hw >> 7) & 1)
        self.Travel = int(device_info[7], 16)
        self.PulsePerPosition = int(device_info[8], 16)
        self.DeviceType = DeviceID.DeviceTypes(int(device_info[2], 16))

        self._set_motor_count()
        self._set_unit_info()

    def _set_motor_count(self):
        device_type = self.DeviceType
        if device_type in {
            DeviceID.DeviceTypes.Shutter2,
            DeviceID.DeviceTypes.Actuator,
        }:
            self.MotorCount = 1
        elif device_type in {
            DeviceID.DeviceTypes.Shutter4,
            DeviceID.DeviceTypes.Shutter6,
            DeviceID.DeviceTypes.Rotator,
            DeviceID.DeviceTypes.RotaryStage8,
            DeviceID.DeviceTypes.LinearStage25mm,
            DeviceID.DeviceTypes.LinearStage28mm,
            DeviceID.DeviceTypes.OpticsRotator,
            DeviceID.DeviceTypes.MotorizedIris,
            DeviceID.DeviceTypes.RotaryStage18,
            DeviceID.DeviceTypes.LinearStage60mm_10,
            DeviceID.DeviceTypes.LinearStage60mm,
        }:
            self.MotorCount = 2
        elif device_type == DeviceID.DeviceTypes.Paddle:
            self.MotorCount = 3
        else:
            self.MotorCount = 1

    def _set_unit_info(self):
        device_type = self.DeviceType
        if device_type == DeviceID.DeviceTypes.Paddle:
            self._unit_factor = (self.PulsePerPosition - 1) / self.Travel
            self._unit_type = DeviceID.UnitTypes.Degrees
            self.FormatStr = '{:.3f}'
        elif device_type in {
            DeviceID.DeviceTypes.Rotator,
            DeviceID.DeviceTypes.OpticsRotator,
            DeviceID.DeviceTypes.RotaryStage8,
            DeviceID.DeviceTypes.RotaryStage18,
        }:
            self._unit_factor = self.PulsePerPosition / 360.0
            self._unit_type = DeviceID.UnitTypes.Degrees
            self.FormatStr = '{:.3f}'
        elif device_type == DeviceID.DeviceTypes.MotorizedIris:
            if self.Imperial:
                self.Travel = self._micron_to_mm(self.Travel)
                self._unit_type = DeviceID.UnitTypes.Inches
                self._unit_factor = 25.4 * self.PulsePerPosition
                self.FormatStr = '{:.4f}'
            else:
                self.Travel = self._micron_to_mm(self.Travel)
                self._unit_type = DeviceID.UnitTypes.MM
                self._unit_factor = self.PulsePerPosition
                self.FormatStr = '{:.2f}'
        else:
            if self.Imperial:
                self._unit_type = DeviceID.UnitTypes.Inches
                self._unit_factor = 25.4 * self.PulsePerPosition
                self.FormatStr = '{:.4f}'
            else:
                self._unit_type = DeviceID.UnitTypes.MM
                self._unit_factor = self.PulsePerPosition
                self.FormatStr = '{:.3f}'

        self.Units = self._unit_type.value

    def pulse_to_unit(self, pulses):
        if self._unit_factor == 0:
            return pulses
        return pulses / self._unit_factor

    def unit_to_pulse(self, units):
        if self._unit_factor == 0:
            return int(units)
        return int(round(units * self._unit_factor))

    def _micron_to_mm(self, micron):
        return micron / 1000.0

    def format_position(self, position, show_units=False, show_space=False):
        output = self.FormatStr.format(position)
        if show_units:
            if show_space:
                output += ' '
            output += self.Units
        return output

    def format_specific_position(
        self, position, decimals, show_units=False, show_space=False
    ):
        format_str = '{:.' + str(decimals) + 'f}'
        output = format_str.format(position)
        if show_units:
            if show_space:
                output += ' '
            output += self.Units
        return output

    def description(self):
        desc = [
            f'Address: {self.Address}',
            f'Serial Number: {self.SerialNo}',
            f'Device Type: {self.DeviceType.name}',
            f'Motors: {self.MotorCount}',
            f'Firmware: {self.Firmware}',
            f'Hardware: {self.Hardware}',
            f"Variant: {'Imperial' if self.Imperial else 'Metric'}",
            f'Year: {self.Year}',
        ]
        if self._unit_type == DeviceID.UnitTypes.MM:
            desc.append(f'Travel: {self.Travel} {self.Units}')
            desc.append(f'Pulses Per {self.Units}: {self.unit_to_pulse(1.0)}')
        elif self._unit_type == DeviceID.UnitTypes.Inches:
            desc.append(f'Travel: {self._micron_to_mm(self.Travel):.3f} {self.Units}')
            desc.append(f'Pulses Per {self.Units}: {self.unit_to_pulse(1.0)}')
        elif self._unit_type == DeviceID.UnitTypes.Degrees:
            desc.append(f'Travel: {self.Travel} {self.Units}')
            desc.append(f'Pulses Per {self.Units}: {self.unit_to_pulse(1.0)}')

        return desc
