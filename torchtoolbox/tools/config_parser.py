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


def circulate_parse(parse_dict, base_path: pathlib.Path, parse_target='yaml'):
    for key, value in parse_dict.copy().items():
        if isinstance(value, str) and value.endswith(f".{parse_target}"):
            sub_config_path = base_path.joinpath(value)
            config = parse_config(sub_config_path)
            if key == '__base__':
                # merge dict
                for sub_key, sub_value in config.items():
                    parse_dict[sub_key] = sub_value
                parse_dict.pop(key)
            else:
                parse_dict[key] = config
        elif isinstance(value, (dict, DotDict)):
            circulate_parse(value, base_path)
