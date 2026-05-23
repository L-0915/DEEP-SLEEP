"""Load YAML config and merge with argparse for DeepSleep training scripts.

YAML structure uses sections for readability:
  training:
    learning_rate: 5e-4

The loader tries both the leaf key ("learning_rate") and the
fully-qualified key ("training_learning_rate") to match argparse args.

Usage:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    ...
    args = parser.parse_args()
    if args.config:
        from configs.config_utils import load_yaml_config
        args = load_yaml_config(args)
"""

import yaml


def _flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_yaml_config(args):
    """Load YAML config and set matching argparse attributes."""
    config_path = getattr(args, "config", None)
    if not config_path:
        return args

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    flat = _flatten(cfg)

    for key, value in flat.items():
        # Try exact match first (e.g. "training_learning_rate")
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            # Try leaf key (e.g. "learning_rate" from "training_learning_rate")
            leaf = key.split("_")[-1] if "_" in key else key
            if hasattr(args, leaf):
                setattr(args, leaf, value)
            else:
                # Try without first section (e.g. "learning_rate" from "model_learning_rate")
                parts = key.split("_")
                for i in range(1, len(parts)):
                    candidate = "_".join(parts[i:])
                    if hasattr(args, candidate):
                        setattr(args, candidate, value)
                        break

    return args
