from typing import Union


class DotDict(dict):
    def __init__(self, data_map: Union[Dict, None] = None):
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
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]
