from enum import Enum


class LaserState(Enum):
    OFF = 'OFF'
    ON = 'ON'
    F1 = 'F1'
    F2 = 'F2'

    def __str__(self):
        return self.value.upper()

    @staticmethod
    def get_list() -> list:
        return [member for member in LaserState]

    @staticmethod
    def get_enum(value: str) -> 'LaserState':
        try:
            return LaserState[value]
        except KeyError:
            return None


class MB_Params(Enum):
    '''
    Enum class defining MatchBox parameters.
    '''
    MODEL = 'Model'
    WAVELENGTH = 'Wavelength'
    WAVELENGTHS = 'Wavelengths [nm]'

    SERIAL_PORT = 'Serial Port'
    PORT = 'Serial Port.Port'
    BAUDRATE = 'Serial Port.Baudrate'
    SET_PORT = 'Serial Port.Set Config'
    OPEN = 'Serial Port.Connect'
    CLOSE = 'Serial Port.Disconnect'
    PORT_STATE = 'Serial Port.State'

    OPTIONS = 'Options'
    SET_CURRENT = 'Options.Set Current'
    STATE = 'Options.State'
    POWER = 'Options.Power [mW]'
    SET_POWER = 'Options.Set Power'

    READINGS = 'Readings'
    POWER_READ = 'Readings.Power [mW]'
    LD_CURRENT = 'Readings.LD Current [mA]'
    LD_CURRENT_SET = 'Readings.LD Set Current [mA]'
    LD_CURRENT_MAX = 'Readings.LD Max Current [mA]'
    LD_TEMP = 'Readings.LD Temp [C]'
    LD_TEMP_SET = 'Readings.LD Set Temp [C]'
    LD_TEC_LOAD = 'Readings.LD TEC Load [%]'
    CRYSTAL_TEMP = 'Readings.Crystal Temp [C]'
    CRYSTAL_TEMP_SET = 'Readings.Crystal Set Temp [C]'
    CRYSTAL_TEC_LOAD = 'Readings.Crystal TEC Load [%]'
    BODY_TEMP = 'Readings.Body Temp [C]'
    FAN_TEMP_SET = 'Readings.Fan Set Temp [C]'
    FAN_LOAD = 'Readings.Fan LOAD [%]'
    STATUS = 'Readings.Status'
    IN_VOLTAGE = 'Readings.Input Voltage [V]'
    AUTO_MODE = 'Readings.Autostart Mode'
    ACCESS_LEVEL = 'Readings.Access Level'
    FEEDBACK_DAC = 'Readings.Feedback DAC'

    INFO = 'Info'
    FIRMWARE = 'Info.Firmware'
    SERIAL = 'Info.Serial Number'
    OPERATION_TIME = 'Info.Operation Time'
    ON_TIMES = 'Info.ON Times'

    REMOVE = 'Remove Device'

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
