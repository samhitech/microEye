import sys
import traceback
import numba

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class thread_worker(QRunnable):
    '''Thread worker

    Inherits from QRunnable to handler worker thread setup,
    signals and wrap-up.

    Parameters
    ----------
    callback : function
        The function callback to run on this worker thread. Supplied args and
        kwargs will be passed through to the runner.
    args:
        Arguments to pass to the callback function
    kwargs:
        Keywords to pass to the callback function
    '''

    def __init__(self, fn, *args, progress=True, z_stage=True, **kwargs):
        super(thread_worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.done = False

        # Add the callback to our kwargs
        if progress:
            self.kwargs['progress_callback'] = self.signals.progress
        if z_stage:
            self.kwargs['movez_callback'] = self.signals.move_stage_z

    @pyqtSlot()
    def run(self):
        '''Initialise the runner function with passed args, kwargs.'''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the processing
            self.signals.result.emit(result)
        finally:
            self.done = True
            self.signals.finished.emit()  # Done


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        No data indicates progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(object)
    move_stage_z = pyqtSignal(bool, int)


def Gaussian(x, a, x0, sigma):
    '''Returns a gaussian

    f(x) = a * exp(-(x - x0)**2 / (2 * sigma**2))

    Parameters
    ----------
    x : ndarray
        input values
    a : float
        amplitude
    x0 : float
        center | mean
    sigma : float
        standard deviation

    Returns
    -------
    ndarray
        gaussian f(x)
    '''
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def GaussianOffSet(x, a, x0, sigma, offset):
    '''Returns a gaussian with offset

    f(x) = a * exp(-(x - x0)**2 / (2 * sigma**2)) + offset

    Parameters
    ----------
    x : ndarray
        input values
    a : float
        amplitude
    x0 : float
        center | mean
    sigma : float
        standard deviation
    offset : float
        y offset

    Returns
    -------
    ndarray
        gaussian f(x)
    '''
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset


def Two_Gaussian(x, a0, x0, sigma0, a1, x1, sigma1):
    '''Returns two gaussians

    f(x) = Gaussian(x, a0, x0, sigma0) +
    Gaussian(x, a1, x1, sigma1)

    Parameters
    ----------
    x : ndarray
        input values
    a0 : float
        amplitude
    x0 : float
        center | mean
    sigma0 : float
        standard deviation
    a1 : float
        amplitude
    x1 : float
        center | mean
    sigma1 : float
        standard deviation

    Returns
    -------
    ndarray
        two gaussians f(x)
    '''
    return Gaussian(x, a0, x0, sigma0) + \
        Gaussian(x, a1, x1, sigma1)


def Three_Gaussian(x, a0, x0, sigma0, a1, x1, sigma1, a2, x2, sigma2, offset):
    '''Returns three gaussians with offset

    f(x) = Gaussian(x, a0, x0, sigma0) +
    Gaussian(x, a1, x1, sigma1) +
    Gaussian(x, a2, x2, sigma2) + offset

    Parameters
    ----------
    x : ndarray
        input values
    a0 : float
        amplitude
    x0 : float
        center | mean
    sigma0 : float
        standard deviation
    a1 : float
        amplitude
    x1 : float
        center | mean
    sigma1 : float
        standard deviation
    a2 : float
        amplitude
    x2 : float
        center | mean
    sigma2 : float
        standard deviation
    offset : float
        y offset

    Returns
    -------
    ndarray
        three gaussians with offset
    '''
    return Gaussian(x, a0, x0, sigma0) + \
        Gaussian(x, a1, x1, sigma1) + \
        Gaussian(x, a2, x2, sigma2) + offset
