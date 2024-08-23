from enum import Enum
from typing import Any, Union

from microEye.qt import QtCore, Signal


class MessageUpdater(QtCore.QObject):
    class UpdateTypes(Enum):
        Status = 1
        MotorInfo = 2
        DeviceInfo = 3
        Position = 4
        HomeOffset = 5
        JogstepSize = 6
        PolarizerPositions = 7
        PaddlePosition = 8

    outputUpdated = Signal(list, bool)
    parameterUpdated = Signal(UpdateTypes, str, object)

    def __init__(self):
        super().__init__()

    def update_output(self, message: Union[str, list[str]], error: bool = False):
        if isinstance(message, str):
            message = [message]
        self.outputUpdated.emit(message, error)

    def update_parameter(self, update_type: UpdateTypes, address: str, data: Any):
        self.parameterUpdated.emit(update_type, address, data)

    def update_output_with_list(self, message: str, list_data: list[str]):
        output_list = [message] + list_data
        self.outputUpdated.emit(output_list, False)
