import sys
import threading
import traceback

import numpy as np

from microEye.qt import QtCore, Signal, Slot


class WorkerSignals(QtCore.QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:
    - finished: No data
    - error: tuple (exctype, value, traceback.format_exc())
    - result: object data returned from processing, anything
    - progress: object for progress updates
    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(object)


class QThreadWorker(QtCore.QRunnable):
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

    def __init__(self, fn, *args, **kwargs):
        super().__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.stop_event = threading.Event()
        self._started = False

        self.kwargs['event'] = self.stop_event

        # Add the callback to our kwargs
        if kwargs.get('progress'):
            self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        '''Initialise the runner function with passed args, kwargs.'''
        self._started = True
        # Retrieve args/kwargs here; and fire processing using them
        try:
            if self.kwargs.get('nokwargs'):
                result = self.fn(*self.args)
            else:
                result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the processing
            self.signals.result.emit(result)
        finally:
            self.stop_event.set()
            self.done = True
            self.signals.finished.emit()  # Done

    def stop(self):
        '''
        Request the worker to stop.
        '''
        self.stop_event.set()

    def is_set(self):
        '''
        Check if the stop has been requested.
        '''
        return self.stop_event.is_set()

    def wait(self, timeout=None):
        '''
        Wait until the worker has finished or the stop has been requested.

        Parameters
        ----------
        timeout : float or None
            The maximum number of seconds to wait. If None, waits indefinitely.
        '''
        if self._started:
            self.stop_event.wait(timeout)
        else:
            raise RuntimeError('Cannot wait on a worker that has not been started')
