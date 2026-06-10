from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yml"


def load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)
