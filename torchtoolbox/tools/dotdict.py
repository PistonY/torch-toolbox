from typing import Dict, Optional


class DotDict(dict):
    def __init__(self, data_map: Optional[Dict] = None):
        if data_map is not None:
            super(DotDict, self).__init__(data_map)
            if isinstance(data_map, dict):
                for k, v in data_map.items():
                    if not isinstance(v, dict):
                        self[k] = v
                    else:
                        self.__setattr__(k, DotDict(v))
        else:
            super().__init__()

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]

    def __str__(self) -> str:
        dump_str = '('
        for k, v in self.__dict__.items():
            dump_str += f"{k}={v}; "
        dump_str = dump_str[:-2] + ")"
        return dump_str

    def pop(self, key):
        value = super().pop(key)
        del self.__dict__[key]
        return value

    @classmethod
    def to_dict(cls, dot_dict):
        new_dict = {}
        for key, value in dot_dict.items():
            if isinstance(value, cls):
                new_dict[key] = cls.as_dict(value)
            else:
                new_dict[key] = value
        return new_dict

    def as_dict(self):
        return self.to_dict(self)
