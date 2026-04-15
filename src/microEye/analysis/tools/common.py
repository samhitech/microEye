import numpy as np


def to_float_or_none(value) -> float | None:
    if value is None:
        return None

    try:
        if isinstance(value, np.ndarray):
            if value.ndim != 0:
                return None
            return float(value.item())
        if isinstance(value, np.generic):
            return float(value.item())
        return float(value)
    except Exception:
        return None


def to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def normalize_positive_float(value, default: float = 1.0) -> tuple[float, bool]:
    parsed = to_float_or_none(value)
    if parsed is None or not (parsed > 0):
        return float(default), True
    return parsed, False
