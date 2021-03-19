from typing import Dict, Optional


class Registry:
    """Provide a string to object transfer.
    Args:
            name (str): name of this registry.
    """
    def __init__(self, name: str) -> None:
        self.name = name
        self._obj_map: Dict(str, object) = {}

    def _do_register(self, name: str, obj: object = None) -> None:
        assert name not in self._obj_map, \
            f"An object named '{name}' was already registered in '{self.name}' registry!"
        self._obj_map[name] = obj

    def register(self, obj: object = None, name: str = None) -> Optional[object]:
        """Register the give object.

        Args:
            obj (object, optional): obj to register. Defaults to None.
            name (str, optional): specific name for this obj.
        Returns:
            Optional[object]: ori obj.
        """
        if obj is None:

            def deco(func_or_class: object) -> object:
                _name = func_or_class.__name__ if name is None else name
                self._do_register(_name, func_or_class)
                return func_or_class

            return deco

        _name = obj.__name__ if name is None else name
        self._do_register(_name, obj)

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f'No object named {name} found in {self.name} register.')
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map
