from typing import Dict, Type, Callable


class Registry:
    def __init__(self):
        self._map: Dict[str, Callable] = {}


    def register(self, name: str):
        def deco(cls_or_fn):
            if name in self._map:
                raise KeyError(f"'{name}' already registered")
            self._map[name] = cls_or_fn
            return cls_or_fn
        return deco


    def get(self, name: str):
        if name not in self._map:
            raise KeyError(f"Unknown registry key: {name}")
        return self._map[name]


MODELS = Registry()
CONDITIONERS = Registry()