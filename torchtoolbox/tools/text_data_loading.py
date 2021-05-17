__all__ = ['load_data', 'merge_data']

import json
import yaml
from .dotdict import DotDict


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
            file = yaml.load(f, **kwargs)
        else:
            raise NotImplementedError
    if to_dot_dict:
        file = DotDict(file)
    return file


def merge_data(dicts, **kwargs):
    new_dict = {}
    for d in dicts:
        new_dict.update(d, **kwargs)
    return new_dict
