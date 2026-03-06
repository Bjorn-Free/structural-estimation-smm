import json
from pathlib import Path


def load_config(path="settings.json"):
    """
    Load configuration settings from a JSON file.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with open(p, "r") as f:
        config = json.load(f)

    return config