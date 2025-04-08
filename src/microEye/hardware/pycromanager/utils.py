from typing import Any


def vec_to_list(vector):
    if hasattr(vector, 'size') and hasattr(vector, 'get'):
        return [vector.get(idx) for idx in range(vector.size())]
    elif hasattr(vector, '__len__') and hasattr(vector, '__getitem__'):
        return [vector[idx] for idx in range(len(vector))]

    raise ValueError('Invalid vector type')


def config_to_dict(config):
    if hasattr(config, 'size') and hasattr(config, 'get_setting'):
        res = {}
        for idx in range(config.size()):
            setting = config.get_setting(idx)
            res[setting.get_key()] = setting.get_property_value()

        return res
    elif isinstance(config, dict):
        return config

    raise ValueError('Invalid config type')


def group_config_dict(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    res = {}

    for key, value in config.items():
        device, prop = key.split('-')
        res.setdefault(device, {})[prop] = value

    return res
