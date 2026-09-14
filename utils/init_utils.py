"""YAML inheritance with explicit command-line overrides."""
import argparse
from pathlib import Path

from omegaconf import OmegaConf


def _load_config(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Cyclic config inheritance: {path}")
    config = OmegaConf.load(path)
    merged = OmegaConf.create()
    for base in config.pop('__base__', []):
        # Existing project configs use repository-relative base paths.
        base_path = Path(base)
        if not base_path.is_absolute() and not base_path.exists():
            base_path = path.parent / base_path
        merged = OmegaConf.merge(merged, _load_config(base_path, (*stack, path)))
    return OmegaConf.merge(merged, config)


def add_args(parser):
    """Precedence: inherited YAML < child YAML < explicit CLI < --set."""
    defaults = {}
    if isinstance(parser, argparse.ArgumentParser):
        parser.add_argument('-c', '--config', default='./config/reproduction.yaml')
        parser.add_argument('--set', nargs='+', default=[])
        # Suppress defaults during parsing, so absent CLI options preserve YAML.
        for action in parser._actions:
            if action.dest != 'help' and action.default != argparse.SUPPRESS:
                defaults[action.dest] = action.default
                action.default = argparse.SUPPRESS
        args = vars(parser.parse_args())
    elif isinstance(parser, argparse.Namespace):
        args = vars(parser).copy()
    else:
        raise TypeError(f'Expected ArgumentParser or Namespace, got {type(parser)}')
    config_path = args.get('config', defaults.get('config'))
    config = _load_config(config_path) if config_path else OmegaConf.create()
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    for key, value in args.items():
        if value is not None:
            config[key] = value
    config['config'] = str(config_path) if config_path else None
    return OmegaConf.merge(config, OmegaConf.from_dotlist(args.get('set', [])))


def config_pretty(config, indent=0):
    print(OmegaConf.to_yaml(config, resolve=True))
