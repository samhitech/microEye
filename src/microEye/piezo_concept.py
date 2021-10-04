from PyQt5.QtSerialPort import *


class piezo_concept(QSerialPort):
    '''PiezoConcept FOC 1-axis stage adapter | Inherits QSerialPort
    '''
    ZPosition = 50000
    LastCmd = ''
    Received = ''

    def GETZ(self):
        '''Gets the current stage position along the Z axis.
        '''
        if(self.isOpen()):
            self.write(b'GET_Z\n')
            self.LastCmd = "GETZ"

    def HOME(self):
        '''Centers the stage position along the Z axis.
        '''
        if(self.isOpen()):
            self.ZPosition = 50000
            self.write(b'MOVEZ 50u\n')
            self.LastCmd = "MOVRZ"

    def NANO_UP(self, step: int):
        '''Moves the stage in the positive direction along the Z axis
        by the specified step in nanometers relative to its last position.
        (Not prefered check UP, DOWN, HOME, REFRESH functions instead)

        Parameters
        ----------
        step : int
            step in nanometers
        '''
        if(self.isOpen()):
            self.write(('MOVRZ +'+step+'n\n').encode('utf-8'))
            self.LastCmd = "MOVRZ"

    def MICRO_UP(self, step: int):
        '''Moves the stage in the positive direction along the Z axis
        by the specified step in micrometers relative to its last position.
        (Not prefered check UP, DOWN, HOME, REFRESH functions instead)

        Parameters
        ----------
        step : int
            step in micrometers
        '''
        if(self.isOpen()):
            self.write(('MOVRZ +'+step+'u\n').encode('utf-8'))
            self.LastCmd = "MOVRZ"

    def NANO_DOWN(self, step: int):
        '''Moves the stage in the negative direction along the Z axis
        by the specified step in nanometers relative to its last position.
        (Not prefered check UP, DOWN, HOME, REFRESH functions instead)

        Parameters
        ----------
        step : int
            step in nanometers
        '''
        if(self.isOpen()):
            self.write(('MOVRZ -'+step+'n\n').encode('utf-8'))
            self.LastCmd = "MOVRZ"

    def MICRO_DOWN(self, step: int):
        '''Moves the stage in the negative direction along the Z axis
        by the specified step in micrometers relative to its last position.
        (Not prefered check UP, DOWN, HOME, REFRESH functions instead)

        Parameters
        ----------
        step : int
            step in micrometers
        '''
        if(self.isOpen()):
            self.write(('MOVRZ -'+step+'u\n').encode('utf-8'))
            self.LastCmd = "MOVRZ"

    def UP(self, step: int):
        '''Moves the stage in the positive direction along the Z axis
        by the specified step in nanometers
        relative to the last position set by the user.

        Parameters
        ----------
        step : int
            step in nanometers
        '''
        if(self.isOpen()):
            self.ZPosition = min(max(self.ZPosition + step, 0), 100000)
            self.write(('MOVEZ '+str(self.ZPosition)+'n\n').encode('utf-8'))
            self.LastCmd = "MOVEZ"

    def DOWN(self, step: int):
        '''Moves the stage in the negative direction
        along the Z axis by the specified step in nanometers
        relative to the last position set by the user.

        Parameters
        ----------
        step : int
            step in nanometers
        '''
        if(self.isOpen()):
            self.ZPosition = min(max(self.ZPosition - step, 0), 100000)
            self.write(('MOVEZ '+str(self.ZPosition)+'n\n').encode('utf-8'))
            self.LastCmd = "MOVEZ"

    def REFRESH(self):
        '''Refresh the stage position
        to the set value in case of discrepancy.
        '''
        if(self.isOpen()):
            self.write(('MOVEZ '+str(self.ZPosition)+'n\n').encode('utf-8'))
            self.LastCmd = "MOVEZ"
