from microEye.qt import QtWidgets


class stage:
    def __init__(self) -> None:
        '''
        Initialize the stage object.

        This method initializes the `stage` object with the starter values.
        '''
        self._connect_btn = QtWidgets.QPushButton()
        self.ZPosition = 50000
        self.LastCmd = ''
        self.Received = ''

    def isOpen(self):
        '''
        Check if the stage is open.

        Returns
        -------
        is_open : bool
            A boolean value indicating whether the stage is open.
        '''
        return False
