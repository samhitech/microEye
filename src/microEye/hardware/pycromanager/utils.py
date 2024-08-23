from typing import Any


def vec_to_list(vector):
    if hasattr(vector, 'size') and hasattr(vector, 'get'):
        return [vector.get(idx) for idx in range(vector.size())]

    return []


def config_to_dict(config):
    res = {}

    if hasattr(config, 'size') and hasattr(config, 'get_setting'):
        for idx in range(config.size()):
            setting = config.get_setting(idx)
            res[setting.get_key()] = setting.get_property_value()

    return res


def group_config_dict(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    res = {}

    for key, value in config.items():
        device, prop = key.split('-')
        res.setdefault(device, {})[prop] = value

    return res
