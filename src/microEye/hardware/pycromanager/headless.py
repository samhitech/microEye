from dataclasses import dataclass
from typing import Optional

import yaml
from mmpycorex.launcher import (
    _JAVA_HEADLESS_SUBPROCESSES,
    _PYMMCORES,
    active_java_ports,
    is_java_port_allocated,
    logger,
    server_terminated,
    terminate_java_instances,
)
from pycromanager.headless import (
    DEFAULT_BRIDGE_PORT,
    start_headless,
    stop_headless,
)


@dataclass
class HeadlessInstance:
    '''Data class to store information about a headless instance.'''

    port: int
    mm_app_path: str
    config_file: Optional[str] = None
    java_loc: Optional[str] = None
    python_backend: bool = False
    core_log_path: str = ''
    buffer_size_mb: int = 1024
    max_memory_mb: int = 2000
    debug: bool = False
    name: str = 'Unnamed Instance'

    def __post_init__(self):
        '''Generate a name if none was provided.'''
        if self.name == 'Unnamed Instance':
            backend = 'Python' if self.python_backend else 'Java'
            self.name = f'{backend}_Backend_{self.port}'


class HeadlessManager:
    '''Manager class for headless Micro-Manager instances.'''

    # Singleton instance
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._instances: dict[int, HeadlessInstance] = {}

    @property
    def instances(self) -> dict[int, HeadlessInstance]:
        '''Get a dictionary of all running instances.'''
        return self._instances.copy()

    @property
    def ports(self) -> list[int]:
        '''Get a list of all ports in use.'''
        return list(self._instances.keys())

    def is_port_used(self, port: int) -> bool:
        '''Check if a port is already in use.'''
        return port in self._instances

    def start_instance(self, instance_config: HeadlessInstance) -> bool:
        '''Start a new headless instance with the given configuration.'''
        if self.is_port_used(instance_config.port):
            return False

        try:
            start_headless(
                mm_app_path=instance_config.mm_app_path,
                config_file=instance_config.config_file,
                java_loc=instance_config.java_loc,
                python_backend=False, # instance_config.python_backend,
                core_log_path=instance_config.core_log_path,
                buffer_size_mb=instance_config.buffer_size_mb,
                max_memory_mb=instance_config.max_memory_mb,
                port=instance_config.port,
                debug=instance_config.debug,
            )
            self._instances[instance_config.port] = instance_config
            return True
        except Exception as e:
            print(f'Failed to start headless instance: {e}')
            return False

    def stop_instance(self, port: int) -> bool:
        '''Stop a specific headless instance by port.'''
        if not self.is_port_used(port):
            return False

        try:
            terminate_java_instances(debug=self._instances[port].debug, port=port)
            del self._instances[port]
            return True
        except Exception as e:
            print(f'Failed to stop headless instance: {e}')
            return False

    def stop_all_instances(self, debug=False) -> None:
        '''Stop all running headless instances.'''
        try:
            terminate_java_instances(debug=debug)
            self._instances.clear()
            return True
        except Exception as e:
            print(f'Failed to stop headless instances: {e}')
            return False

    def get_instance(self, port: int) -> Optional[HeadlessInstance]:
        '''Get instance information for a specific port.'''
        return self._instances.get(port)

    def save_config(self, filepath: str) -> bool:
        '''Save the current configuration to a file.'''
        # Implementation for saving configs to JSON/YAML
        try:
            with open(filepath, 'w') as f:
                yaml.dump(self._instances, f)
            return True
        except Exception as e:
            print(f'Failed to save configuration: {e}')
            return False

    def load_config(self, filepath: str) -> bool:
        '''Load configuration from a file.'''
        # Implementation for loading configs from JSON/YAML
        # Each instance will be a dictionary with the port as the key
        # They should be started and added if not confilicting with existing ones
        try:
            with open(filepath) as f:
                _instances = yaml.load(f, Loader=yaml.FullLoader)

            for port, instance in _instances.items():
                if not self.is_port_used(port):
                    self.start_instance(HeadlessInstance(**instance))

            return True
        except Exception as e:
            print(f'Failed to load configuration: {e}')
            return False
