__all__ = ['load_data', 'save_data']

import json
import yaml
from ..tools import DotDict


def load_data(file_path: str, format: str, to_dot_dict: bool = False, load_kwargs: dict = dict(mode='r'), **kwargs) -> dict:
    """Use this func to load file easily.

    Args:
        file_path (str): full file path
        format (str): parse engine
        to_dot_dict (bool, optional): whether convert to dot_dict. Defaults to False.
        load_kwargs (dict, optional): this will give `open()`. Defaults to {}.

    Raises:
        NotImplementedError: [description]

    Returns:
        dict: [description]
    """
    assert format in ('json', 'yaml'), 'Now only json and yaml format are supported.'
    with open(file_path, **load_kwargs) as f:
        if format == 'json':
            file = json.load(f, **kwargs)
        elif format == 'yaml':
            file = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            raise NotImplementedError
    if to_dot_dict:
        file = DotDict(file)
    return file


def save_data(data, file_path, format, load_kwargs: dict = dict(mode='w'), **kwargs) -> None:
    assert format in ('json', 'yaml'), 'Now only json and yaml format are supported.'
    with open(file_path, **load_kwargs) as f:
        if format == 'json':
            json.dump(data, f, indent=2, **kwargs)
        elif format == 'yaml':
            yaml.dump(data, f, Dumper=yaml.SafeDumper, **kwargs)
        else:
            raise NotImplementedError
