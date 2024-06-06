import sys
import traceback

import numpy as np

from microEye.qt import QtCore, Signal, Slot


class thread_worker(QtCore.QRunnable):
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
        super().__init__()

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

    @Slot()
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


class WorkerSignals(QtCore.QObject):
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
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(object)
    move_stage_z = Signal(bool, int)
