import threading
from abc import ABC, abstractmethod


class Device(ABC):
    def __init__(self):
        super().__init__()
        self._event = threading.Event()

    def wait_for_event(self, timeout=None):
        '''
        Wait for an event to occur.

        When the timeout argument is present and not None, it should be a floating point
        number specifying a timeout for the operation in seconds (or fractions thereof).
        '''
        self._event.wait(timeout=timeout)

    def set_event(self):
        '''Set the event to trigger waiting threads.'''
        self._event.set()

    def clear_event(self):
        '''Reset the event to allow waiting again.'''
        self._event.clear()

    @abstractmethod
    def open(self):
        '''Open the hardware.'''
        pass

    @abstractmethod
    def close(self):
        '''Close the hardware.'''
        pass

    @abstractmethod
    def get_property(self, property_name: str):
        '''Get the value of a property.'''
        pass

    @abstractmethod
    def set_property(self, property_name: str, value):
        '''Set the value of a property.'''
        pass

    @abstractmethod
    def emit_signal(self, signal):
        '''Emit a signal.'''
        pass

    def __init_subclass__(cls):
        abstract_methods = [
            m for m in cls.__abstractmethods__ if m not in (
                '__abstractmethods__', '__subclasshook__')]
        if abstract_methods:
            raise TypeError(
                f'Cannot instantiate abstract class {cls.__name__} ' + \
                    f'with abstract methods {", ".join(abstract_methods)}')
        super().__init_subclass__()
