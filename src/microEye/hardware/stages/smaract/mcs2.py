import logging

from microEye.hardware.stages.stage import AbstractStage
from microEye.qt import QtWidgets

try:
    import smaract.ctl as ctl
except ImportError:
    ctl = None


def assert_lib_compatibility():
    '''
    Checks that the major version numbers of the Python API and the
    loaded shared library are the same to avoid errors due to
    incompatibilities.
    Raises a RuntimeError if the major version numbers are different.
    '''
    if ctl is None:
        raise ImportError('SmarActCTL library not found.')

    vapi = ctl.api_version
    vlib = [int(i) for i in ctl.GetFullVersionString().split('.')]
    if vapi[0] != vlib[0]:
        raise RuntimeError('Incompatible SmarActCTL python api and library version.')


class MCS2Stage(AbstractStage):
    NAME = 'SmarAct MCS2'

    def __init__(self, locator):
        super().__init__(
            name=MCS2Stage.NAME,
            # max_range=(0, 0, max_um * 1000),
            # units=Units.NANOMETERS,
            readyRead=None,
        )

        if locator is None:
            raise ValueError('MCS2 locator must be specified.')

        self.__handle = None
        self.locator = locator

    def is_open(self):
        return ctl is not None and self.__handle is not None

    def open(self):
        if not self.is_open():
            locators = MCS2Stage.find_devices()
            if locators and self.locator in locators:
                try:
                    self.__handle = ctl.Open(self.locator)
                    logging.info(f'MCS2 connected to {self.locator}.')
                except Exception as e:
                    self.__handle = None
                    logging.error(f'MCS2 failed to connect to {self.locator}. {e}')
            else:
                logging.error(f'MCS2 device {self.locator} not found.')

    @classmethod
    def find_devices(cls):
        '''
        Finds available MCS2 devices to connect to.

        Returns
        -------
        list[str]
            List of available device locators.
        '''
        try:
            assert_lib_compatibility()

            buffer = ctl.FindDevices()
            if len(buffer) == 0:
                logging.warning('MCS2 no devices found.')
                raise ConnectionError
            locators = buffer.split('\n')
            for locator in locators:
                logging.info(f'MCS2 available devices: {locator}')
        except Exception as e:
            locators = []
            logging.error(f'MCS2 failed to find devices. {e}')

        return locators.copy()

    @classmethod
    def get_stage(cls, **kwargs):
        locators = cls.find_devices()
        if len(locators) == 0:
            return None
        if len(locators) == 1:
            return cls(locator=locators[0], **kwargs)
        else:
            # If there are multiple instances, prompt the user to select one
            locator, ok = QtWidgets.QInputDialog.getItem(
                None,
                'Select MCS2 Device',
                'Select the MCS2 device to use:',
                locators,
            )
            if ok and locator:
                return cls(locator=locator, **kwargs)
            else:
                return None

    def get_config(self) -> dict:
        config = super().get_config()

        config['locator'] = self.locator

        return config

    def load_config(self, config: dict):
        if not isinstance(config, dict):
            raise ValueError('Config must be a dictionary.')

        super().load_config(config)

        locator = config.get('locator')
        if locator is not None:
            self.close()
            self.locator = locator
