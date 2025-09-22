import traceback
from typing import Optional

from pyqtgraph.parametertree import Parameter

from microEye.hardware.stages.kinesis.kdc101 import KDC101Controller
from microEye.hardware.stages.stage import (
    AbstractStage,
    Axis,
    StageDriver,
    StageParams,
    Units,
    emit_after_signal,
)
from microEye.qt import QtCore, QtSerialPort, QtWidgets, Signal
from microEye.utils.parameter_tree import Tree
from microEye.utils.thread_worker import QThreadWorker


class KinesisXY(AbstractStage):
    '''Class for controlling Two Kinesis Devices as an XY Stage'''

    NAME = 'Kinesis 2 x (KDC101 + Z825B)'

    def __init__(self, x_port='COM12', y_port='COM11', **kwargs):
        super().__init__(
            name=KinesisXY.NAME,
            max_range=kwargs.get('max_range', (25, 25)),
            units=Units.MILLIMETERS,
            axes=(Axis.X, Axis.Y),
            readyRead=None,
            center=kwargs.get('center', [17, 13]),
        )

        self.X = KDC101Controller(portName=x_port)
        self.Y = KDC101Controller(portName=y_port)
        self.prec = 4
        self._driver = StageDriver.DUAL_SERIALPORT

    @property
    def x(self) -> float:
        pos = self.X.position
        self.set_position(axis=Axis.X, position=pos)
        return pos

    @property
    def y(self) -> float:
        pos = self.Y.position
        self.set_position(axis=Axis.Y, position=pos)
        return pos

    def home(self, is_async=True):
        self.run_async(self.X.home, is_async=is_async)
        self.run_async(self.Y.home, is_async=is_async)

    @emit_after_signal('moveFinished')
    def move_absolute(self, x, y, z=0, **kwargs):
        force = kwargs.get('force', False)
        is_async = kwargs.get('is_async', True)

        x = round(max(0, min(self.x_max, x)), self.prec)
        y = round(max(0, min(self.y_max, y)), self.prec)
        if x != self.X.position or force:
            self.run_async(self.X.move_absolute, x, is_async=is_async)
        if y != self.Y.position or force:
            self.run_async(self.Y.move_absolute, y, is_async=is_async)

    @emit_after_signal('moveFinished')
    def move_relative(self, x, y, z=0, **kwargs):
        is_async = kwargs.get('is_async', True)

        x = round(x, self.prec)
        y = round(y, self.prec)
        if x != 0:
            self.run_async(self.X.move_relative, x, is_async=is_async)
        if y != 0:
            self.run_async(self.Y.move_relative, y, is_async=is_async)

    def refresh_position(self):
        self.move_absolute(self.X.position, self.Y.position, force=True, is_async=True)

    def stop(self):
        self.X.move_stop()
        self.Y.move_stop()

    def open(self):
        '''Opens the serial ports.'''
        res = self.X.open() and self.Y.open()
        return res

    def close(self):
        '''Closes the serial ports.'''
        self.X.close()
        self.Y.close()

    def run_async(self, callback, *args, is_async=True):
        if self.is_open() and not self._busy:
            self.X.clearResponses()
            self.Y.clearResponses()

            def worker_callback():
                try:
                    self._busy = True
                    self.signals.asyncStarted.emit()
                    callback(*args)
                except Exception as e:
                    traceback.print_exc()
                finally:
                    self._busy = False
                    self.signals.asyncFinished.emit()

            if is_async:
                _worker = QThreadWorker(worker_callback, nokwargs=True)

                QtCore.QThreadPool.globalInstance().start(_worker)
            else:
                worker_callback()

    def is_open(self):
        return all([self.X.isOpen(), self.Y.isOpen()])

    def get_config(self) -> dict:
        config = super().get_config()

        config['X']['port'] = self.X.portName()
        config['X']['baudrate'] = self.X.baudRate()

        config['Y']['port'] = self.Y.portName()
        config['Y']['baudrate'] = self.Y.baudRate()

        return config

    def load_config(self, config: dict):
        if not isinstance(config, dict):
            raise ValueError('Config must be a dictionary.')

        super().load_config(config)

        x_config = config.get('X', {})
        y_config = config.get('Y', {})

        if isinstance(x_config, dict):
            self.X.setBaudRate(x_config.get('baudrate', 115200))
            self.X.setPortName(x_config.get('port', 'COM12'))

        if isinstance(y_config, dict):
            self.Y.setBaudRate(y_config.get('baudrate', 115200))
            self.Y.setPortName(y_config.get('port', 'COM13'))


if __name__ == '__main__':
    XY = KinesisXY('COM11', 'COM12')

    # print(XY.home())
    print(XY.center())

    # for i in range(10):
    #     k = 0.01
    #     XY.move_relative(k, k)
    #     sleep(1)
    # for i in range(10):
    #     k = - 0.01
    #     XY.move_relative(k, k)
    #     sleep(1)
