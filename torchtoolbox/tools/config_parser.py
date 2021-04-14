from .dotdict import DotDict
import yaml
import pathlib


def parse_config(config_file: str):
    config_file = pathlib.Path(config_file)
    assert config_file.suffix in ('.yml', '.yaml'), "Only support yaml files."
    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg = DotDict(cfg)
    circulate_parse(cfg, config_file.parent)
    return cfg


def merge_dict(target_dict: dict, sub_dict: dict, key: str, replace=False):
    """merge sub dict to target dict.

    Now special key is only `__base__`, this can be added if needed.

    Args:
        target_dict (dict): target to merge
        sub_dict (dict): merge this to target
        key (str): key word
        replace (bool, optional): Whether replace if target sub key is not None. Defaults to False.
    """
    if key == '__base__':
        for sub_key, sub_value in sub_dict.items():
            if sub_key not in target_dict.keys() or replace:
                target_dict[sub_key] = sub_value
        target_dict.pop(key)

    else:
        target_dict[key] = sub_dict


def circulate_parse(parse_dict, base_path: pathlib.Path, parse_target='yaml'):
    for key, value in parse_dict.copy().items():
        if isinstance(value, str) and value.endswith(f".{parse_target}"):
            sub_config_path = base_path.joinpath(value).resolve()
            config = parse_config(sub_config_path)
            merge_dict(parse_dict, config, key)
        elif isinstance(value, (dict, DotDict)):
            circulate_parse(value, base_path)
