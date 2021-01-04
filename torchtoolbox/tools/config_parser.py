from .dotdict import DotDict
import yaml


def parse_config(config_file: str):
    assert config_file.endswith('.yml') or config_file.endswith('.yaml'), \
        "Only support yaml files."
    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg = DotDict(cfg)
    return cfg
