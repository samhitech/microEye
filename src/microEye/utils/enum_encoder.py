import json
from enum import Enum


# Custom encoder for JSON serialization
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return str(obj)
        return super().default(obj)
