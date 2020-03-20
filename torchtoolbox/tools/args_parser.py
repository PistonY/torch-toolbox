# -*- coding: utf-8 -*-
__all__ = ['JsonParse', 'ArgsParse']

import json


def load_json(json_path):
    with open(self.file_path, 'r') as f:
        js = json.load(f)
    assert len(js) > 0, 'Json file is empty.'
    return js

class Parse(object):
    """
    A parser to support configuration file.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.file_ret = None
        self.__load_file__()
        assert self.file_ret is not None
        self.__parse_file__()

    def __load_file__(self):
        raise NotImplementedError

    def __repr__(self):
        format_string = 'Namespace{}'.format(self.js).replace('{', '(').replace('}', ')')
        return format_string

    def __parse_file__(self):
        raise NotImplementedError

    def format_str(self, string):
        return string.lower().replace('-', '_').replace(' ', '')


class JsonParse(Parse):
    """A json file parser.
    """

    def __init__(self, file_path):
        super(JsonParse, self).__init__(file_path)

    def __load_file__(self):
        self.file_ret = load_json(self.file_path)

    def __parse_file__(self):
        for k, v in self.js.items():
            k = self.format_str(k)
            if not isinstance(v, dict):
                setattr(self, k, v)
            else:
                assert 'value' in v.keys(), \
                    "Need `value` in second level attributes."
                if v['value'] is not None:
                    setattr(self, k, v['value'])
                    continue
                assert 'default' in v.keys(), \
                    'If value is `None` need a default value.'
                setattr(self, k, v['default'])
