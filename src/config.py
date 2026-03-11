import copy
import json
from pathlib import Path


def _deep_update(base: dict, updates: dict) -> dict:
    """
    Recursively update a nested dictionary.

    Values in `updates` overwrite values in `base`. If both values are
    dictionaries, update recursively.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _apply_validation_overrides(config: dict) -> dict:
    """
    Apply runtime overrides from config["debug_overrides"][validation_mode]
    when debug_mode is enabled.

    This centralizes the effective runtime configuration so that all modules
    see the same simulation and DP settings.
    """
    config = copy.deepcopy(config)

    debug_mode = bool(config.get("debug_mode", False))
    validation_mode = str(config.get("validation_mode", "fast"))

    if not debug_mode:
        return config

    debug_overrides = config.get("debug_overrides", {})
    if not isinstance(debug_overrides, dict):
        return config

    mode_overrides = debug_overrides.get(validation_mode, {})
    if not isinstance(mode_overrides, dict):
        return config

    _deep_update(config, mode_overrides)
    return config


def _validate_config(config: dict) -> dict:
    """
    Validate core configuration fields and add a few safe defaults.
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary.")

    config.setdefault("debug_mode", False)
    config.setdefault("validation_mode", "fast")
    config.setdefault("rebuild_data", False)
    config.setdefault("theta0", [0.1, 0.1, 0.8, 0.25])
    config.setdefault(
        "bounds",
        [
            [0.01, 1.0],
            [0.1, 15.0],
            [0.6, 0.95],
            [0.1, 0.5],
        ],
    )
    config.setdefault("n_moments", 9)

    config.setdefault("model", {})
    config.setdefault("simulation", {})
    config.setdefault("dp", {})
    config.setdefault("debug_overrides", {})

    required_top_level = [
        "raw_data_path",
        "clean_data_path",
    ]
    missing = [key for key in required_top_level if key not in config]
    if missing:
        raise KeyError(f"Missing required config entries: {missing}")

    theta0 = config.get("theta0")
    if not isinstance(theta0, list) or len(theta0) != 4:
        raise ValueError("config['theta0'] must be a list of length 4.")

    bounds = config.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ValueError("config['bounds'] must be a list of length 4.")

    for i, bound in enumerate(bounds):
        if not isinstance(bound, list) or len(bound) != 2:
            raise ValueError(f"config['bounds'][{i}] must be a list of length 2.")
        lower, upper = bound
        if lower >= upper:
            raise ValueError(
                f"config['bounds'][{i}] must satisfy lower < upper, got {bound}."
            )

    n_moments = int(config.get("n_moments", 0))
    if n_moments <= 0:
        raise ValueError("config['n_moments'] must be positive.")

    return config


def load_config(path="settings.json"):
    """
    Load project configuration from JSON, then apply effective runtime settings.

    Behavior
    --------
    1. Load the raw JSON file.
    2. Validate the basic structure.
    3. If debug_mode is True, apply debug_overrides[validation_mode].
    4. Validate the final effective config.
    5. Store the original raw config under '_raw_config' for reference.

    This ensures every downstream module sees one coherent effective config.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    raw_config = _validate_config(raw_config)
    effective_config = _apply_validation_overrides(raw_config)
    effective_config = _validate_config(effective_config)

    effective_config["_raw_config"] = copy.deepcopy(raw_config)

    return effective_config