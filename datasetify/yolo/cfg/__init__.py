from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from datasetify.utils import yaml_load, DEFAULT_CFG_PATH

def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object to be converted to a dictionary.

    Returns:
        cfg (dict): Configuration object in dictionary format.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_yolo_cfg(cfg=DEFAULT_CFG_PATH):
    """Get YOLO config from path"""

    cfg = cfg2dict(cfg)

    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])

    return Namespace(**cfg)
